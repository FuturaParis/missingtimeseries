import csv
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn import svm, metrics, preprocessing, neighbors
import copy
 

KKColList =   ['M_DSPLY_SAL',
               'M_EXP_VAL_SAL_CST_MUSD',
               'M_EXP_VAL_SAL_CST_NON_PROMO',
               'M_EXP_VAL_SAL_CST_PROMO',
               'M_EXP_VOL_SAL_MSU',
               'M_EXP_VOL_SAL_NON_PROMO',
               'M_EXP_VOL_SAL_PROMO',
               'M_FEATR_SAL',
               'M_TDP',
               'M_TDP_DSPLY',
               'M_TDP_FEATR',
               'M_WD_DSPLY',
               'M_WD_DSPLY_FEATR',
               'M_WD_FEATR',
               'M_WGHT_DISTRIB']

def sum_by_group(values, groups):
    order = np.argsort(groups)
    groups = groups[order]
    values = values[order]
    values.cumsum(out=values, axis=0)
    index = np.ones(len(groups), 'bool')
    index[:-1] = groups[1:] != groups[:-1]
    values = values[index]
    groups = groups[index]
    values[1:] = values[1:] - values[:-1]
    return values, groups

def ColNum(ColNameList,Array):
    ColList = []
    for i, column in enumerate(Array[0,:]):
        if column in ColNameList:
            ColList.append(i)
    if len(ColList) == 1:
        ColList = ''.join(str(ColList)[1:-1])
    return ColList

def Select(ColNameList, Array):
    return Array[:,ColNum(ColNameList,Array)]

def Distinct(Array):
    TupledArray = [tuple(row) for row in Array]
    UniqueTuples = list(set(TupledArray))
    return np.array(UniqueTuples) 

def BorderZeroGenerator(Array,ProductSet,TimeLvlSetSort,MeasuresList,TimeCol,IDCol,Reversed=False):
    CompleteCnt = 0
    InCompleteCnt = 0
    IDTimeArray = Distinct(Select(IDCol + TimeCol, Array))
    if Reversed:
        TimeLvlSetSort.reverse()
    for product in ProductSet:
        #check how many time levels exist for product
        ProductTimeLvlSet = IDTimeArray[IDTimeArray[:,0] == product]
        CompleteCnt += 1
        #if anything is missing proceed
        if ProductTimeLvlSet.shape[0] != len(TimeLvlSetSort):
            CompleteCnt -= 1
            InCompleteCnt += 1
            NotFirstFoundFlag = True
            FirstEmptyPeriodList = []
            #check time levels from the beginning
            for period in TimeLvlSetSort:
                if ProductTimeLvlSet[ProductTimeLvlSet[:,1] == period].shape[0] != 0:
                    NotFirstFoundFlag = False
                    RowForCopy = Array[(Array[:,ColNum([TimeCol], Array)] == period)*(Array[:,ColNum([IDCol], Array)] == product)]
                    FirstIter = True
                    for MissingPeriod in FirstEmptyPeriodList:
                        RowToInsert = copy.deepcopy(RowForCopy)
                        RowToInsert[:,ColNum([TimeCol],Array)] = MissingPeriod
                        for MeasureToNull in MeasuresList:
                            RowToInsert[:,ColNum([MeasureToNull], Array)] = '0'
                        if FirstIter:
                            ArrayToInsert = RowToInsert
                            FirstIter = False
                        else:
                            ArrayToInsert = np.vstack((ArrayToInsert, RowToInsert))
                    break
                else:
                    #not found any value yet, add period to list to later fill with null row
                    FirstEmptyPeriodList.append(period)
                    print "For %s, period %s not found" % (product, period)


        if InCompleteCnt > 10:
            break
    return ArrayToInsert

def InsideZeroGenerator(Array,ProductSet,TimeLvlSetSort,MeasuresList,TimeCol,IDCol):
    CompleteCnt = 0
    InCompleteCnt = 0
    IDTimeArray = Distinct(Select(IDCol + TimeCol, Array))
    for product in ProductSet:
        #check how many time levels exist for product
        ProductTimeLvlSet = IDTimeArray[IDTimeArray[:,0] == product]
        CompleteCnt += 1
        #if anything is missing proceed
        if ProductTimeLvlSet.shape[0] != len(TimeLvlSetSort):
            CompleteCnt -= 1
            InCompleteCnt += 1
            NotFirstFoundFlag = True
            PeriodMissed = False
            FirstEmptyPeriodList = []
            #check time levels from the beginning
            for period in TimeLvlSetSort:
                #found time period
                if ProductTimeLvlSet[ProductTimeLvlSet[:,1] == period].shape[0] != 0 and not PeriodMissed:
                    NotFirstFoundFlag = False
                    #copy measure values
                    RowForCopy = Array[(Array[:,ColNum([TimeCol], Array)] == period)*(Array[:,ColNum([IDCol], Array)] == product)]
                    SavedMeasures = Select(MeasuresList, RowForCopy)
                    FirstIter = True
                elif ProductTimeLvlSet[ProductTimeLvlSet[:,1] == period].shape[0] = 0 and not NotFirstFoundFlag and not PeriodMissed:
                    PeriodMissed = True
                elif PeriodMissed:
                    if ProductTimeLvlSet[ProductTimeLvlSet[:,1] == period].shape[0] = 0

                    for MissingPeriod in FirstEmptyPeriodList:
                        RowToInsert = copy.deepcopy(RowForCopy)
                        RowToInsert[:,ColNum([TimeCol],Array)] = MissingPeriod
                        for MeasureToNull in MeasuresList:
                            RowToInsert[:,ColNum([MeasureToNull], Array)] = '0'
                        if FirstIter:
                            ArrayToInsert = RowToInsert
                            FirstIter = False
                        else:
                            ArrayToInsert = np.vstack((ArrayToInsert, RowToInsert))
                    break
                else:
                    #not found any value yet, add period to list to later fill with null row
                    FirstEmptyPeriodList.append(period)
                    print "For %s, period %s not found" % (product, period)


        if InCompleteCnt > 10:
            break
    return ArrayToInsert

def NullFactGenerator(Array,MeasuresList,TimeCol,IDCol):
    ProductSet = Select([IDCol],Array)[1:,]
    TimeLvlSet = np.unique(Select([TimeCol],Array)[1:,])
    TimeLvlSet = [datetime.strptime(period, '%m/%y') for period in TimeLvlSet]
    TimeLvlSet.sort()
    TimeLvlSetSort = [period.strftime('%m/%y').lstrip('0') for period in TimeLvlSet]

    ArrayToInsertFor = BorderZeroGenerator(Array,ProductSet,TimeLvlSetSort,MeasuresList,TimeCol,IDCol,Reversed=False)
    ArrayToInsertRev = BorderZeroGenerator(Array,ProductSet,TimeLvlSetSort,MeasuresList,TimeCol,IDCol,Reversed=True)

    ArrayToInsertBorder = np.vstack((ArrayToInsertFor, ArrayToInsertRev))






 
    



with open("Data - competition (2).csv",'r') as dest_f:
    data_iter = csv.reader(dest_f, delimiter = ',')
    data = [data for data in data_iter]


data_array = np.array(data) 


NullFactGenerator(data_array,KKColList,'T_TIME_NAME','PROD_ID')
##print data_array.shape
##data_array = data_array[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,33,34]]
##
##DistinctTimeLvl = set(data_array[1:,ColNum(['T_TIME_NAME'],data_array)])
##Cnt = 0
##
##for time in DistinctTimeLvl:
##    print time
##    TimeSubset = data_array[data_array[:,ColNum(['T_TIME_NAME'],data_array)] == time]
##    #ProdSubset = TimeSubset[1:,ColNum(ColNum(['PROD_ID'],data_array),data_array)]
##    TimeSubset = TimeSubset[1:,ColNum(KKColList,TimeSubset)]
##    nbrs = neighbors.NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(TimeSubset)
##    distances, indices = nbrs.kneighbors(TimeSubset)
##    if Cnt != 0:
##        try:
##            ResultArray = np.vstack((ResultArray, indices[:,1]))
##        except:
##           #print ResultArray.shape
##           print indices[:,1].shape 
##    else:
##        ResultArray = indices[:,1]
##    Cnt =+ 1
##print ResultArray
#Price = data_array[1:,0]
#MeanPrice = np.mean(Price[Price!='unknown'].astype(np.float), axis=0)
#Price[Price=='unknown'] = MeanPrice
#
#PriceScaled = preprocessing.scale(Price.astype(np.float))
#PriceScaled = np.expand_dims(PriceScaled, axis=1)

#target = data_array[1:,4]
#dataset = np.hstack((PriceScaled, data_array[1:,5:]))


#NSample = len(dataset)
#
#np.random.seed(34567)
#order = np.random.permutation(NSample)
#
#dataset = dataset[order]
#target = target[order].astype(np.float)
#
#
#dataset_train = dataset[:.9 * NSample]
#target_train = target[:.9 * NSample]
#dataset_test = dataset[.9 * NSample:]
#target_test = target[.9 * NSample:]

#targetCont = np.expand_dims(target, axis=1)
#Cvalues, Cgroups = sum_by_group(dataset[:,1:].astype(int), target.astype(int))


#np.savetxt("ContTable.txt", Cvalues, delimiter=";")

