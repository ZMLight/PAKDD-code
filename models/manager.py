#! -*- coding:utf-8 -*-

from models.ModelA0 import Model as A0
from models.ModelA1 import Model as A1
from models.ModelA2 import Model as A2
from models.ModelA3 import Model as A3
from models.ModelA3_CNN import Model as A3_CNN
from models.ModelA4 import Model as A4
from models.ModelA3_CNN_ATT import Model as A3_CNN_ATT
from models.ModelA3_CNN_ATT2 import Model as A3_CNN_ATT2

import sys
def get_model2(args,W1,W2):
	if args.nn_model=='A0':			return A0(args)
	if args.nn_model=='A1':			return A1(args)
	if args.nn_model=='A2':			return A2(args)
	if args.nn_model=='A3':			return A3(args)
	if args.nn_model=='A3_CNN':		return A3_CNN(args)
	if args.nn_model=='A3_CNN_ATT':	return A3_CNN_ATT(args,W1,W2)
	if args.nn_model=='A3_CNN_ATT2':	return A3_CNN_ATT2(args,W1,W2)
	if args.nn_model=='A4':			return A4(args)
	print('no such model:',args.nn_model)
	sys.exit(1)

def get_model(args):
	if args.nn_model=='A0':			return A0(args)
	if args.nn_model=='A1':			return A1(args)
	if args.nn_model=='A2':			return A2(args)
	if args.nn_model=='A3':			return A3(args)
	if args.nn_model=='A3_CNN':		return A3_CNN(args)
	if args.nn_model=='A3_CNN_ATT':	return A3_CNN_ATT(args)
	if args.nn_model=='A4':			return A4(args)
	print('no such model:',args.nn_model)
	sys.exit(1)
