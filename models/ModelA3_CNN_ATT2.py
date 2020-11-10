#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

from collections import defaultdict
import sys,random
import math


class AttentionLayer(chainer.Chain):
	def __init__(self, dim, dropout_rate):
		super(AttentionLayer, self).__init__(
			x2z	=L.Linear(3*dim,dim),
			z2b =L.Linear(dim,1),
			#x2z =L.Convolution2D(1,1,2),
			#bn	=L.BatchNormalization(1)
		)
		self.dropout_rate=dropout_rate
	def __call__(self, x):
		if self.dropout_rate!=0:
			x = F.dropout(x,ratio=self.dropout_rate)
		z = self.x2z(x)
		z = self.z2b(z)
		z = F.leaky_relu(z)

		return z

class Module(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, isR, isBN):
		super(Module, self).__init__(
			x2z	=L.Linear(2*dim,dim),
			#x2z =L.Convolution2D(1,1,2),
			bn	=L.BatchNormalization(dim)
		)
		self.dropout_rate=dropout_rate
		self.activate = activate
		self.is_residual = isR
		self.is_batchnorm = isBN
	def __call__(self, x):
		if self.dropout_rate!=0:
			x = F.dropout(x,ratio=self.dropout_rate)
		z = self.x2z(x)
		#z = F.reshape(z,(-1,200))

		#if self.is_batchnorm:
			#z = self.bn(z)
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
		linksAtt = [('a{}'.format(i), AttentionLayer(dim,dropout_rate)) for i in range(1)]
		for link in linksAtt:
			self.add_link(*link)
		self.AttL = linksAtt

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



	"""
	# neighbor_entities=[(k,v)]
	# (e,k) in links
	# e = entities[i]
	# i in assing[v]
	"""
	"""
	source entityから出てるedgeが無い
	"""
	def __call__(self,x,rel_y,neighbor_entities,neighbor_dict,assign,entities,relations,RC,EC,t,assignEtoN):
		if self.layer==0:
			return self.easy_case(x,neighbor_entities,neighbor_dict,assign,entities,relations)


		#print 'entities', len(entities)
		if len(neighbor_dict)==1:
			x=[x]
		else:
			x = F.split_axis(x,len(neighbor_dict),axis=0)

		if len(entities) == 1:
			t = [t]
		else:
			t = F.split_axis(t,len(entities),axis=0)

		rel_y = F.split_axis(rel_y,len(RC),axis=0)

		result = []
		for i,e in enumerate(entities):
			rt = t[i]
			tmpXList = []
			tmpValList = []

			tmpListV1 = []
			tmpListV2 = []
			tmpList2 = []
			tmpVFlag = []

			for k in assignEtoN[i]:
				v = neighbor_dict[k]
				rx = x[v]

				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				r_rep = rel_y[r//2]

				#calc the attention value
				tmp2 = F.concat((rx,rt),axis=1)
				tmp2 = F.concat((tmp2,r_rep),axis=1)
				tmpList2.append(tmp2)

				tmp = F.concat((rx,r_rep),axis=1)
				#tmp = F.pad(F.concat((rx,r_rep),axis=0), ((0,0),(0,1)), 'constant')
				#tmp = F.reshape(tmp,(1,1,2,-1))

				if r%2==0:
					tmpListV1.append(tmp)
					tmpVFlag.append(1)
				else:
					tmpListV2.append(tmp)
					tmpVFlag.append(-1)
			#print len(tmpListV1), len(tmpListV2), len(tmpList2)

			oV1 = []
			oV2 = []
			oAtt = []

			if(len(tmpListV1) > 0):
				inputV1 = F.concat(tmpListV1,axis=0)
				#print inputV1.shape
				outputV1 = getattr(self,self.forwardH[0][0])(inputV1)
				#print outputV1.shape
				oV1 = F.split_axis(outputV1, len(tmpListV1), axis = 0)

			if(len(tmpListV2) > 0):
				inputV2 = F.concat(tmpListV2,axis=0)
				outputV2 = getattr(self,self.forwardT[0][0])(inputV2)
				oV2 = F.split_axis(outputV2, len(tmpListV2), axis = 0)

			inputAtt = F.concat(tmpList2,axis=0)
			#print inputAtt.shape
			outputAtt = getattr(self,self.AttL[0][0])(inputAtt)
			#print outputAtt.shape
			oAtt = F.split_axis(outputAtt, len(tmpList2), axis = 0)

			cnt1 = 0
			cnt2 = 0

			for a,flag in enumerate(tmpVFlag):
				tmpAtt = oAtt[a]
				tmpAtt = F.repeat(tmpAtt, 200)
				tmpAtt = F.reshape(tmpAtt, [-1,200])
				tmpValList.append(F.exp(tmpAtt))
				if flag == 1:
					tmprx = oV1[cnt1]
					cnt1 += 1
					tmpXList.append(tmprx)
				elif flag == -1:
					tmprx = oV2[cnt2]
					cnt2 += 1
					tmpXList.append(tmprx)

			#print len(tmpXList), len(tmpValList)

			for a,val in enumerate(tmpValList):
				#print tmpXList[a].data
				#print val.data
				tmpXList[a] = tmpXList[a] * val
			result.append(sum(tmpXList)/(sum(tmpValList)))
			#print result[0].shape #(1,1,1,200) should be (1,200)

		result = F.concat(result,axis=0)
		#print len(entities)
		#print result.shape
		return result



class Model(chainer.Chain):
	def __init__(self, args, W1, W2):
		super(Model, self).__init__(
			embedE	= L.EmbedID(args.entity_size,args.dim, initialW = W2),
			embedE2 = L.EmbedID(args.entity_size,args.dim, initialW = W2),
			embedR	= L.EmbedID(args.rel_size,args.dim, initialW = W1),
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

	def Normal(self,xp):

		tmplist = [1]

		tmp1 = self.embedE(xp.array(tmplist,'i'))
		print(tmp1.data)

		allE = list()
		for i in range(self.e_size):
			allE.append(i)
		allEembed = self.embedE(xp.array(allE,'i'))
		#print allEembed.data
		allEembed = F.normalize(allEembed)

		tmp2 = self.embedE(xp.array(tmplist,'i'))
		print(tmp2.data)

		allR = list()
		for i in range(self.r_size):
			allR.append(i)
		allRembed = self.embedR(xp.array(allR,'i'))
		#print allRembed.data
		allRembed = F.normalize(allRembed)
		#print allRembed.shape
		return 0

	def get_context(self,entities,links,relations,edges,order,xp,RC,EC):
		if self.depth==order:
			return self.embedE(xp.array(entities,'i'))

		assign = defaultdict(list)
		assignEtoN = defaultdict(list)
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
				if len(edges[e])<=self.sample_size:	nn = edges[e]
				else:		nn = random.sample(edges[e],self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in edges',e,self.is_known,order)
					sys.exit(1)
			nn = list(nn)
			assignEtoN[i] = nn
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
		t = self.embedE(xp.array(entities,'i'))

		x = getattr(self,self.forwardB[order][0])(x,rel_y,neighbor,neighbor_dict,assign,entities,relations,RC,EC,t,assignEtoN)
		return x

	def train(self,positive,negative,links,relations,edges,xp,RC,RCT,EC,ECT):
		self.cleargrads()
		entities= set()
		for h,r,t in positive:
			entities.add(h)
			entities.add(t)
		for h,r,t in negative:
			entities.add(h)
			entities.add(t)
		entities = list(entities)

		x = self.get_context(entities,links,relations,edges,0,xp,RC,EC)
		x = F.split_axis(x,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,x):
			edict[e]=x

		pos,rels = [],[]
		for h,r,t in positive:
			rels.append(r)
			pos.append(edict[h]-edict[t])
		pos = F.concat(pos,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		#pos = F.sum(F.absolute(pos+xr),axis=1)
		pos = F.batch_l2_norm_squared(pos+xr)

		neg,rels = [],[]
		for h,r,t in negative:
			rels.append(r)
			neg.append(edict[h]-edict[t])
		neg = F.concat(neg,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		#neg = F.sum(F.absolute(neg+xr),axis=1)
		neg = F.batch_l2_norm_squared(neg+xr)

		if self.objective_function=='relative': return sum(F.relu(self.threshold+pos-neg)),pos
		if self.objective_function=='absolute': return sum(pos+F.relu(self.threshold-neg)),pos


	def get_scores(self,candidates,links,relations,edges,xp,mode,RC,EC):
		entities = set()
		for h,r,t,l in candidates:
			entities.add(h)
			entities.add(t)
		entities = list(entities)
		xe = self.get_context(entities,links,relations,edges,0,xp,RC,EC)
		xe = F.split_axis(xe,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,xe):
			edict[e]=x
		diffs,rels = [],[]
		for h,r,t,l in candidates:
			rels.append(r)
			diffs.append(edict[h]-edict[t])
		diffs = F.concat(diffs,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		scores = F.batch_l2_norm_squared(diffs+xr)
		return scores

	def write_vec(self,args,links,relations,edges,xp,RC,EC):
		xe = self.get_context(range(args.entity_size),links,relations,edges,0,xp,RC,EC)

		f1 = open(args.trained_vec+"entity2vec.vec",'w')
		for i in range(args.entity_size):
			for j in range(args.dim):
				f1.write(str(xe.data[i][j]) + '\t')
			f1.write('\n')
		f1.close()

		xr = self.embedR(xp.array(range(args.rel_size),'i'))
		f2 = open(args.trained_vec+"relation2vec.vec",'w')
		for i in range(args.rel_size):
			for j in range(args.dim):
				f2.write(str(xr.data[i][j]) + '\t')
			f2.write('\n')
		f2.close()


	def get_scores3(self,candidates,links,relations,edges,xp,mode,RC,EC):
		h,r,t,l = candidates
		xe1 = self.embedE2(xp.array(range(self.e_size),'i'))
		xe2 = self.embedE2(xp.array([t],'i'))
		xe2 = F.broadcast_to(xe2,(self.e_size,200))
		xe3 = self.embedE2(xp.array([h],'i'))
		xe3 = F.broadcast_to(xe3,(self.e_size,200))
		xr = self.embedR(xp.array([r],'i'))
		xr = F.broadcast_to(xr,(self.e_size,200))
		if self.is_bound_wr:	xr = F.tanh(xr)
		scores1 = F.batch_l2_norm_squared(xe1-xe2+xr)
		scores2 = F.batch_l2_norm_squared(xe3-xe1+xr)
		return scores1,scores2
