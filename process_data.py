from pyspi.calculator import CalculatorFrame, CorrelationFrame
import _pickle as cPickle

with open('data/db.pkl','rb') as f:
    database = cPickle.load(f)

names = list(database.keys())

# To test
names = names[:3]

datasets = [database[n]['data'].T for n in names]
labels = [database[n]['labels'] for n in names]

calcf = CalculatorFrame(datasets=datasets,labels=labels,names=names,fast=True)
calcf.compute()

corrf = CorrelationFrame(calcf)
mm_adj = corrf.get_average_correlation()
print(mm_adj)