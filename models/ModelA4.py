#!/usr/bin/env python
# -*- coding: utf-8 -*-
# try 3NN model, add relation prediction part 11/21

import chainer
import chainer.functions as F
import chainer.links as L

from collections import defaultdict
import sys,random

class Module(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, isR, isBN):
		super(Module, self).__init__(
			#x2z	=L.Linear(2*dim,dim),
			x2z = L.Convolution2D(1,1,2),
			bn	=L.BatchNormalization(1)
		)
		self.dropout_rate=dropout_rate
		self.activate = activate
		self.is_residual = isR
		self.is_batchnorm = isBN
	def __call__(self, x):
		if self.dropout_rate!=0:
			x = F.dropout(x,ratio=self.dropout_rate)
		z = self.x2z(x)
		if self.is_batchnorm:
			z = self.bn(z)
		if self.activate=='tanh': z = F.tanh(z)
		if self.activate=='relu': z = F.relu(z)
		if self.is_residual:	return z+x
		else: return z
class Block(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN):
		super(Block, self).__init__()
		links = [('m{}'.format(i), Module(dim,dropout_rate,activate, isR, isBN)) for i in range(layer)]
		for link in links:
			self.add_link(*link)
		self.forward = links
	def __call__(self,x):
		for name, _ in self.forward:
			x = getattr(self,name)(x)
		return x

class Tunnel(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN, relation_size, pooling_method):
		super(Tunnel, self).__init__()
		linksH = [('h{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(1)]
		for link in linksH:
			self.add_link(*link)
		self.forwardH = linksH
		linksT = [('t{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(1)]
		for link in linksT:
			self.add_link(*link)
		self.forwardT = linksT
		linksR = [('r{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(1)]
		for link in linksR:
			self.add_link(*link)
		self.forwardR = linksR
		self.pooling_method = pooling_method
		self.layer = layer

	def maxpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:
				x = F.concat(xxs,axis=0)					# -> (b,d)
				x = F.swapaxes(x,0,1)						# -> (d,b)
				x = F.maxout(x,len(xxs))					# -> (d,1)
				x = F.swapaxes(x,0,1)						# -> (1,d)
				result.append(x)
		return result

	def averagepooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs)/len(xxs))
		return result

	def valpooling(self,xs,neighbor,RC,EC):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result=[]
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs)/len(xxs))
		return result

	def sumpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs))
		return result

	def easy_case(self,x,neighbor_entities,neighbor_dict,assign,entities,relations):
		x = F.split_axis(x,len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				assignR[(r,len(bundle[r]))] = v
				bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		for r in bundle:
			rx = bundle[r]
			if len(rx)==1:	result[assignR[(r,0)]] = rx[0]
			else:
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		return result


	def case_for_relation(self,neighborWeight,neighbor,assign,Rs,relationsT):

		neighborR = list()
		#print len(neighbor)
		for i,t in enumerate(neighbor):
			#print t.shape
			t = F.reshape(t,(2,-1))
			t = F.pad(t, ((0,0),(0,1)), 'constant')
			t = F.reshape(t,(1,1,2,-1))
			#print t.shape
			t = getattr(self,self.forwardR[0][0])(t)
			#print t.shape

			t = F.reshape(t,(1,-1))
			#print t.shape
			neighborR.append(t)

		resultT = list()
		for i,r in enumerate(Rs):
			Rlist = assign[i]
			templist = list()
			sumWeight = 0
			for x in Rlist:
				templist.append(neighborR[x]*neighborWeight[x])
				sumWeight = sumWeight + neighborWeight[x]
			resultT.append(sum(templist)/sumWeight)

		result = F.concat(resultT,axis=0)
		return result

	"""
	# neighbor_entities=[(k,v)]
	# (e,k) in links
	# e = entities[i]
	# i in assing[v]
	"""
	"""
	source entityから出てるedgeが無い
	"""
	def __call__(self,x,rel_y,neighbor_entities,neighbor_dict,assign,entities,relations,RC,EC,flag):
		if(flag==False):
			return self.case_for_relation(rel_y,neighbor_entities,assign,entities,relations)
		if self.layer==0:
			return self.easy_case(x,neighbor_entities,neighbor_dict,assign,entities,relations)

		if len(neighbor_dict)==1:
			x=[x]
		else:
			x = F.split_axis(x,len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				assignR[(r,len(bundle[r]))] = v
				bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		rel_y = F.split_axis(rel_y,len(RC),axis=0)
		for r in bundle:
			rx = bundle[r]
			try:
				r_rep = rel_y[r//2]
			except:
				print r//2

			if len(rx)==1:
				rx=rx[0]
				tmp = F.pad(F.concat((rx,r_rep),axis=0),((0,0),(0,1)),'constant')
				tmp = F.reshape(tmp,(1,1,2,-1))
				if r%2==0:	rx = getattr(self,self.forwardH[0][0])(tmp)
				else:		rx = getattr(self,self.forwardT[0][0])(tmp)
				rx = F.reshape(rx,(1,-1))
				result[assignR[(r,0)]] = rx
			else:
				size = len(rx)
				#tempRx = list()
				for i in range(size):
					rx[i] = F.pad(F.concat((rx[i],r_rep),axis=0), ((0,0),(0,1)), 'constant')
				rx = F.concat(rx,axis=0)

				tmp = F.reshape(rx, (size,1,2,-1))
				if r%2==0:	rx = getattr(self,self.forwardH[0][0])(tmp)
				else:		rx = getattr(self,self.forwardT[0][0])(tmp)

				rx = F.reshape(rx,(size,-1))
				rx = F.split_axis(rx,size,axis=0)
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		#if self.pooling_method=='val':
		#	result = self.valpooling(result,assign,RC,EC)

		if self.pooling_method=='weight':
			sources = defaultdict(list)
			for ee in assign:
				for i in assign[ee]:
					sources[i].append((ee,result[ee]))
			resultT=[]
			for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
				ii = entities[i]
				if len(xxs)==1:
					resultT.append(xxs[0][1])
				else:
					tmplist = []
					for txxs in xxs:
						k = txxs[0]
						ke = txxs[1]
						'''
						if (ii,k) in relations:
							r = relations[(ii,k)]
						else if (k,ii) in relations:
							r = relations[(k,ii)]'''
						tmplist.append(EC[k])
					tmplist2 = []
					if(sum(tmplist) != 0.0):
						for t in range(len(xxs)):
							tmplist2.append(xxs[t][1]*tmplist[t])
						resultT.append(sum(tmplist2)/sum(tmplist))
					else:
						for t in range(len(xxs)):
							tmplist2.append(xxs[t][1])
						resultT.append(sum(tmplist2)/len(xxs))
			result = resultT

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)

		result = F.concat(result,axis=0)
		return result



class Model(chainer.Chain):
	def __init__(self, args):
		super(Model, self).__init__(
			embedE	= L.EmbedID(args.entity_size,args.dim),
			embedR	= L.EmbedID(args.rel_size,args.dim),
		)
		linksB = [('b{}'.format(i), Tunnel(args.dim,args.dropout_block,args.activate,args.layerR,args.is_residual,args.is_batchnorm, args.rel_size, args.pooling_method)) for i in range(args.order)]
		for link in linksB:
			self.add_link(*link)
		self.forwardB = linksB

		self.e_size = args.entity_size
		self.r_size = args.rel_size
		self.sample_size = args.sample_size
		self.dropout_embed = args.dropout_embed
		self.dropout_decay = args.dropout_decay
		self.depth = args.order
		self.is_embed = args.is_embed
		self.is_known = args.is_known
		self.threshold = args.threshold
		self.objective_function = args.objective_function
		self.is_bound_wr = args.is_bound_wr
		if args.use_gpu: self.to_gpu()

	def get_context_relation(self,Rs,links,relationsT,edges,order,xp,RC,EC,relationsEdges):
		assign = defaultdict(list)
		neighbor_dict = defaultdict(int)
		for i,r in enumerate(Rs):
			if r in relationsT:
				if(len(relationsT[r])<=self.sample_size): nn = relationsT[r]
				else:		nn = random.sample(relationsT[r],self.sample_size)
				if len(nn) == 0:
					print('something wrong @ modelS')
					print('relation not in relationsT',r)
					sys.exit(1)
			else:
				if len(relationsEdges[r])<=self.sample_size:	nn = relationsEdges[r]
				else:		nn = random.sample(relationsEdges[r],self.sample_size)
				if len(nn) == 0:
					print('something wrong @ modelS')
					print('relation not in relationsEdges',r)
					sys.exit(1)

			for k in nn:
				if k not in neighbor_dict:
					neighbor_dict[k] = len(neighbor_dict) #(k,v)
				assign[i].append(neighbor_dict[k])

		neighbor = []
		neighborWeight =[]
		for k,v in sorted(neighbor_dict.items(),key=lambda x:x[1]):
			templist = list()
			templist.append(k[0])
			templist.append(k[1])
			tempht = self.embedE(xp.array(templist,'i'))
			tempht = F.split_axis(tempht,len(templist),axis=0)
			tempht1 = F.concat((tempht[0],tempht[1]),axis=1)
			#print tempht1.shape
			neighbor.append(tempht1)
			neighborWeight.append(EC[k[0]]+EC[k[1]])

		x = getattr(self,self.forwardB[order][0])(neighborWeight,neighborWeight,neighbor,neighbor_dict,assign,Rs,relationsT,RC,EC,False)
		return x

	def get_context(self,entities,links,relations,edges,order,xp,RC,EC):
		if self.depth==order:
			return self.embedE(xp.array(entities,'i'))

		assign = defaultdict(list)
		neighbor_dict = defaultdict(int)
		for i,e in enumerate(entities):
			"""
			(not self.is_known)
				unknown setting
			(not is_train)
				in test time
			order==0
				in first connection
			"""
			if e in links:
				if len(links[e])<=self.sample_size:	nn = links[e]
				else:		nn = random.sample(links[e],self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in links',e,self.is_known,order)
					sys.exit(1)
			else:
				print('entity not in links')
				sys.exit(1)
				'''
				if len(edges[e])<=self.sample_size:	nn = edges[e]
				else:		nn = random.sample(edges[e],self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in edges',e,self.is_known,order)
					sys.exit(1)'''
			for k in nn:
				if k not in neighbor_dict:
					neighbor_dict[k] = len(neighbor_dict)	# (k,v)
				assign[neighbor_dict[k]].append(i)
		neighbor = []
		for k,v in sorted(neighbor_dict.items(),key=lambda x:x[1]):
			neighbor.append(k)
		x = self.get_context(neighbor,links,relations,edges,order+1,xp,RC,EC)

		rel_y_l = list()
		for i in range(self.r_size):
			rel_y_l.append(i)
		rel_y = self.embedR(xp.array(rel_y_l,'i'))
		x = getattr(self,self.forwardB[order][0])(x,rel_y,neighbor,neighbor_dict,assign,entities,relations,RC,EC,True)
		return x

	def train(self,positive,negative,links,relations,edges,xp,RC,RCT,EC,ECT,relationsT,relationsEdges):
		self.cleargrads()
		entities= set()
		Rs = set()
		for h,r,t in positive:
			entities.add(h)
			entities.add(t)
			Rs.add(r)
		for h,r,t in negative:
			entities.add(h)
			entities.add(t)
			Rs.add(r)
		entities = list(entities)
		Rs = list(Rs)

		x = self.get_context(entities,links,relations,edges,0,xp,RC,EC)
		x = F.split_axis(x,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,x):
			edict[e]=x

		RsE = self.get_context_relation(Rs,links,relationsT,edges,0,xp,RC,EC,relationsEdges)
		RsE = F.split_axis(RsE,len(Rs),axis=0)
		rdict = dict()
		for r,x in zip(Rs,RsE):
			rdict[r]=x

		pos,rels = [],[]
		for h,r,t in positive:
			rels.append(rdict[r])
			pos.append(edict[h]-edict[t])
		pos = F.concat(pos,axis=0)
		xr = F.concat(rels,axis=0)
		#xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		#pos = F.sum(F.absolute(pos+xr),axis=1)
		pos = F.batch_l2_norm_squared(pos+xr)

		neg,rels = [],[]
		for h,r,t in negative:
			rels.append(rdict[r])
			neg.append(edict[h]-edict[t])
		neg = F.concat(neg,axis=0)
		xr = F.concat(rels,axis=0)
		#xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		#neg = F.sum(F.absolute(neg+xr),axis=1)
		neg = F.batch_l2_norm_squared(neg+xr)

		if self.objective_function=='relative': return sum(F.relu(self.threshold+pos-neg)),pos
		if self.objective_function=='absolute': return sum(pos+F.relu(self.threshold-neg)),pos


	def get_scores(self,candidates,links,relations,edges,xp,mode,RC,EC,relationsT,relationsEdges):
		entities = set()
		Rs = set()
		for h,r,t,l in candidates:
			entities.add(h)
			entities.add(t)
			Rs.add(r)
		entities = list(entities)
		xe = self.get_context(entities,links,relations,edges,0,xp,RC,EC)
		xe = F.split_axis(xe,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,xe):
			edict[e]=x

		Rs = list(Rs)
		xr = self.get_context_relation(Rs,links,relationsT,edges,0,xp,RC,EC,relationsEdges)
		xr = F.split_axis(xr,len(Rs),axis=0)
		rdict=dict()
		for r,x in zip(Rs,xr):
			rdict[r] = x

		diffs,rels = [],[]
		for h,r,t,l in candidates:
			rels.append(rdict[r])
			diffs.append(edict[h]-edict[t])
		diffs = F.concat(diffs,axis=0)
		xr = F.concat(rels,axis=0)
		#xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		scores = F.batch_l2_norm_squared(diffs+xr)
		return scores
