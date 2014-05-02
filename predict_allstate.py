import MySQLdb,csv
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import tree
from  sklearn import feature_selection  

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

#http://blog.yhathq.com/posts/logistic-regression-and-python.html

def getdb():
	return MySQLdb.connect("localhost", "root", "lumos123", "allstate")
	
def prepare_trainset():
	print "prepare train"
	feat_gen = open("train_set.csv","w+")
	feat_gen.write("cid,grp_size,homeowner,car_val,risk_factor,married,c_prev,age_diff,a_rt,b_rt,c_rt,d_rt,e_rt,f_rt,g_rt,avg_a,avg_b,avg_c,avg_d,avg_e,avg_f,avg_g,cost,a,b,c,d,e,f,g\n")
	#query = "SELECT s.cid,grp_size,homeowner,car_val,(((car_age-u.avg_age)/(max_age-min_age))*risk_factor*grp_size) as ip,married,c_prev,age_oldest,age_youngest,a,b,c,d,e,f,g,avg_a,avg_b,avg_c,avg_d,avg_e,avg_f,avg_g,cover_cost from train_dataset s, (SELECT cid,round(avg(a)) as avg_a,round(avg(b)) as avg_b,round(avg(c)) as avg_c,round(avg(d)) as avg_d,round(avg(e)) as avg_e,round(avg(f)) as avg_f,round(avg(g)) as avg_g,avg(cost) as cover_cost from train_dataset where rec_type=0 group by cid) t,(SELECT avg(car_age) as avg_age,max(car_age) as max_age,min(car_age) as min_age FROM train_dataset) u  where s.cid=t.cid and rec_type=1"
	query = "SELECT s.cid,grp_size,homeowner,car_val,(((car_age-u.avg_age)/(max_age-min_age))*risk_factor*grp_size) as ip,married,c_prev,age_oldest,age_youngest,a,b,c,d,e,f,g,avg_a,avg_b,avg_c,avg_d,avg_e,avg_f,avg_g,cover_cost from train_dataset s, (SELECT a.cid,a as avg_a,b as avg_b,c as avg_c,d as avg_d,e as avg_e,f as avg_f,g as avg_g,cover_cost from train_dataset a,(SELECT cid,max(shop_pt)-1 as max_pt,avg(cost) as cover_cost from train_dataset where rec_type=0 group by cid) b where a.cid=b.cid and shop_pt=max_pt ) t,(SELECT avg(car_age) as avg_age,max(car_age) as max_age,min(car_age) as min_age FROM train_dataset) u  where s.cid=t.cid and rec_type=1"
	db = getdb()
	cursor = db.cursor()
	lines = cursor.execute(query)
	rset = cursor.fetchall()

	prod_options = ['a','b','c','d','e','f','g']
	
	prod_rsets = {}
	print "prod options"
	for prod in prod_options:
		print "generating retain prob for option"+prod
		query = "SELECT cid,count(*),"+prod+" from train_dataset where rec_type=0 group by cid,"+prod
		cursor = db.cursor()
		lines = cursor.execute(query)
		rest = cursor.fetchall()
		id = rest[0][0]
		max = 0
		tot = 0
		num = 0
		cids_retain = {}
		for r in rest:
			curr_id = r[0]
			if id!=curr_id:
				#cids_retain[id] = round(float(max)/float(tot),2)
				cids_retain[id] = num
				max = r[1]
				num = r[2]
				tot = 0
				id = curr_id
			else:
				if r[1]>max:
					max = r[1]
					num = r[2]
			tot += r[1]
		#cids_retain[id] = round(float(max)/float(tot),2)
		cids_retain[id] = num
		prod_rsets[prod] = cids_retain
		
	for r in rset:
		cid = r[0]
		print "customer--->"+str(cid)
		#car value
		car_val = r[3]
		if car_val == "a":
			car_val = 1
		elif car_val == "b":
			car_val = 2
		elif car_val == "c":
			car_val = 3
		elif car_val == "d":
			car_val = 4
		elif car_val == "e":
			car_val = 5
		elif car_val == "f":
			car_val = 6
		elif car_val == "g":
			car_val = 7
		elif car_val == "h":
			car_val = 8
		elif car_val == "i":
			car_val = 9
			
		#risk factor
		risk_fac = r[4]
		'''
		if risk_fac == 1:
			risk_fac = 1
		elif risk_fac == 2: 
			risk_fac = 0.5
		elif risk_fac == 3: 
			risk_fac = 0.25
		elif risk_fac == 4: 
			risk_fac = 0
		'''
		
		age_old = r[7]
		age_yg = r[8]
		age_diff = round(float(age_old-age_yg)/float(age_old),2)
		
		#c_prev
		c_prev = r[6]
		if c_prev == r[18]:
			c_prev_retain = 1
		else:
			c_prev_retain = 0
		
		
		feat_gen.write(str(cid)+","+str(r[1])+","+str(r[2])+","+str(car_val)+","+str(risk_fac)+","+str(r[5])+","+str(c_prev_retain)+","+str(age_diff)+","+str(prod_rsets["a"][cid])+","+str(prod_rsets["b"][cid])+","+str(prod_rsets["c"][cid])+","+str(prod_rsets["d"][cid])+","+str(prod_rsets["e"][cid])+","+str(prod_rsets["f"][cid])+","+str(prod_rsets["g"][cid])+","+str(r[16])+","+str(r[17])+","+str(r[18])+","+str(r[19])+","+str(r[20])+","+str(r[21])+","+str(r[22])+","+str(r[23])+","+str(r[9])+","+str(r[10])+","+str(r[11])+","+str(r[12])+","+str(r[13])+","+str(r[14])+","+str(r[15])+"\n")
	feat_gen.close()
	
def feature_select(df):
	prod_options = ['a','b','c','d','e','f','g']
	selected_features = {}
	
	for opt in prod_options:
		feats = feature_selection.SelectKBest(k="all")
		sets = convert_df(df,opt)
		preds =  np.asarray(df[opt],dtype=np.float32)
		labels = []
		for p in preds:
			labels.append(int(p))
		sets = feats.fit(sets,labels)
		selected_features[opt] = feats
		
	return selected_features

def check_correlation(df):
	corr_arrays = {15:[16,17,18,19,20,21],16:[17,18,19,20,21],17:[18,19,20,21],18:[19,20,21],19:[20,21],20:[21]}
	
	for corr in corr_arrays:
		others = corr_arrays[corr]
		val1 = np.array(df[[df.columns[corr]]])
		print "correlation against the attrib : "+str(corr)
		for attr in others:
			val2 = np.array(df[[df.columns[attr]]])
			res = stats.pearsonr(val1,val2)			
			print "corr["+str(corr)+","+str(attr)+"]::"+str(res)
	
def cross_validate_model(k=10):
	df = pd.read_csv("train_set.csv")
	check_correlation(df)
	
	train_cols =  df.columns[4:]
	#print df["pred"]
	print "training the model"
	tot_len = len(df)
	eq_size = tot_len/k
	print tot_len
	start = 0
	end = eq_size
	
	
	
	test_data = df[start:end]
	train_data = df[end:tot_len]
	
	for i in range(1,k+1):
		print "Fold--->"+str(i)
			
		models = get_train_models(train_data)	
		tot_len = len(test_data)
		
		matched = 0
		
		a_sets = convert_df(test_data,"a")
		b_sets = convert_df(test_data,"b")
		c_sets = convert_df(test_data,"c")
		d_sets = convert_df(test_data,"d")
		e_sets = convert_df(test_data,"e")
		f_sets = convert_df(test_data,"f")
		g_sets = convert_df(test_data,"g")
		
		ct = 0
		for item in test_data.iterrows():	

			pred_a = models["a"]["classifier"].predict(models["a"]["feat_sel"].transform(a_sets[ct]))
			pred_b = models["b"]["classifier"].predict(models["b"]["feat_sel"].transform(b_sets[ct]))
			pred_c = models["c"]["classifier"].predict(models["c"]["feat_sel"].transform(c_sets[ct]))
			pred_d = models["d"]["classifier"].predict(models["d"]["feat_sel"].transform(d_sets[ct]))
			pred_e = models["e"]["classifier"].predict(models["e"]["feat_sel"].transform(e_sets[ct]))
			pred_f = models["f"]["classifier"].predict(models["f"]["feat_sel"].transform(f_sets[ct]))
			pred_g = models["g"]["classifier"].predict(models["g"]["feat_sel"].transform(g_sets[ct]))
			
			ct = ct+1
			opt_vectors = str(pred_a[0])+str(pred_b[0])+str(pred_c[0])+str(pred_d[0])+str(pred_e[0])+str(pred_f[0])+str(pred_g[0])
			org_vectors = str(int(item[1]["a"]))+str(int(item[1]["b"]))+str(int(item[1]["c"]))+str(int(item[1]["d"]))+str(int(item[1]["e"]))+str(int(item[1]["f"]))+str(int(item[1]["g"]))
			
			if opt_vectors == org_vectors:
				matched += 1
		
		print "Accuracy--->"+str(float(matched)/float(tot_len))
		
		start = end
		end = end+eq_size
		
		test_data = df[start:end]	
		train_data = [df[0:start] , df[end:tot_len]]
		train_data = pd.concat(train_data)

def convert_df(df,opt):
	
	#cid,grp_size,homeowner,car_val,risk_factor,married,c_prev,age_diff,a_rt,b_rt,c_rt,d_rt,e_rt,f_rt,g_rt,avg_a,avg_b,avg_c,avg_d,avg_e,avg_f,avg_g,cost,a,b,c,d,e,f,g
	
	if opt == "a":
		train_cols =  [df.columns[3],df.columns[4],df.columns[15]]
	elif opt == "b":
		train_cols =  [df.columns[3],df.columns[4],df.columns[16]]
	elif opt == "c":
		train_cols =  [df.columns[3],df.columns[4],df.columns[17]]
	elif opt == "d":
		train_cols =  [df.columns[3],df.columns[4],df.columns[18]]
	elif opt == "e":
		train_cols =  [df.columns[3],df.columns[4],df.columns[19]]
	elif opt == "f":
		train_cols =  [df.columns[3],df.columns[4],df.columns[20]]
	elif opt == "g":
		train_cols =  [df.columns[3],df.columns[4],df.columns[21]]
	
	'''
	if opt == "a":
		train_cols =  [df.columns[3],df.columns[4],df.columns[15],df.columns[16],df.columns[17],df.columns[18],df.columns[19],df.columns[20],df.columns[21]]
	elif opt == "b":
		train_cols =  [df.columns[3],df.columns[4],df.columns[16],df.columns[17],df.columns[18],df.columns[19],df.columns[20],df.columns[21]]
	elif opt == "c":
		train_cols =  [df.columns[3],df.columns[4],df.columns[17],df.columns[18],df.columns[19],df.columns[20],df.columns[21]]
	elif opt == "d":
		train_cols =  [df.columns[3],df.columns[4],df.columns[18],df.columns[19],df.columns[20],df.columns[21]]
	elif opt == "e":
		train_cols =  [df.columns[3],df.columns[4],df.columns[19],df.columns[20],df.columns[21]]
	elif opt == "f":
		train_cols =  [df.columns[3],df.columns[4],df.columns[20],df.columns[21]]
	elif opt == "g":
		train_cols =  [df.columns[3],df.columns[4],df.columns[21]]
	'''
	'''
	if opt == "a":
		train_cols =  [df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[7],df.columns[8],df.columns[15]]
	elif opt == "b":
		train_cols =  [df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[7],df.columns[9],df.columns[16]]
	elif opt == "c":
		train_cols =  [df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[6],df.columns[7],df.columns[10],df.columns[17]]
	elif opt == "d":
		train_cols =  [df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[7],df.columns[11],df.columns[18]]
	elif opt == "e":
		train_cols =  [df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[7],df.columns[12],df.columns[19]]
	elif opt == "f":
		train_cols =  [df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[7],df.columns[13],df.columns[20]]
	elif opt == "g":
		train_cols =  [df.columns[1],df.columns[2],df.columns[3],df.columns[4],df.columns[5],df.columns[7],df.columns[14],df.columns[21]]
	'''
	return np.array(df[train_cols])
		
#7 different classifiers to predict the options
def get_train_models(df):
	train_models = {}
	prod_options = ['a','b','c','d','e','f','g']
	
	print "Training..."
	
	#Feature Selection
	feat_sels = feature_select(df)		
	
	for opt in prod_options:
		sets = convert_df(df,opt)
		
		sets = feat_sels[opt].transform(sets)
		
		#print sets
		preds =  np.asarray(df[opt],dtype=np.float32)
		labels = []
		for p in preds:
			labels.append(int(p))
							
		#classifier = tree.DecisionTreeClassifier(criterion="entropy")
		classifier = RandomForestClassifier(n_estimators=10)
		#classifier = LogisticRegression()
		
		result = classifier.fit(sets,labels)
		class_feats = {}
		class_feats["classifier"] = result
		class_feats["feat_sel"] = feat_sels[opt]
		train_models[opt] = class_feats
		
	return train_models
		
def train_classifier():
	df = pd.read_csv("train_set.csv")
	return get_train_models(df)
	
#to be changed accordingly to train dataset yet
def predict_test(models):
	feat_gen = open("final_submit.csv","w+")
	feat_gen.write("customer_ID,plan\n")
	
	#query = "SELECT s.cid,grp_size,homeowner,car_val,(((car_age-u.avg_age)/(max_age-min_age))*risk_factor*grp_size) as ip,married,c_prev,age_oldest,age_youngest,a,b,c,d,e,f,g,avg_a,avg_b,avg_c,avg_d,avg_e,avg_f,avg_g,cover_cost from test_dataset s, (SELECT cid,round(avg(a)) as avg_a,round(avg(b)) as avg_b,round(avg(c)) as avg_c,round(avg(d)) as avg_d,round(avg(e)) as avg_e,round(avg(f)) as avg_f,round(avg(g)) as avg_g,max(shop_pt)as max_pt,avg(cost) as cover_cost from test_dataset group by cid) t, (SELECT avg(car_age) as avg_age,max(car_age) as max_age,min(car_age) as min_age FROM train_dataset) u  where s.cid=t.cid and s.shop_pt=max_pt"
	query = "SELECT s.cid,grp_size,homeowner,car_val,(((car_age-u.avg_age)/(max_age-min_age))*risk_factor*grp_size) as ip,married,c_prev,age_oldest,age_youngest,a,b,c,d,e,f,g,a,b,c,d,e,f,g,cover_cost from test_dataset s, (SELECT cid,max(shop_pt)as max_pt,avg(cost) as cover_cost from test_dataset group by cid) t, (SELECT avg(car_age) as avg_age,max(car_age) as max_age,min(car_age) as min_age FROM train_dataset) u  where s.cid=t.cid and s.shop_pt=max_pt"
	db = getdb()
	cursor = db.cursor()
	lines = cursor.execute(query)
	rset = cursor.fetchall()
	prod_options = ['a','b','c','d','e','f','g']
	
	prod_rsets = {}
	print "predicting..."
	for prod in prod_options:
		print "generating retain prob for option"+prod
		query = "SELECT cid,count(*) from test_dataset group by cid,"+prod
		cursor = db.cursor()
		lines = cursor.execute(query)
		rest = cursor.fetchall()
		id = rest[0][0]
		max = 0
		tot = 0
		cids_retain = {}
		for r in rest:
			curr_id = r[0]
			if id!=curr_id:
				cids_retain[id] = round(float(max)/float(tot),2)
				max = r[1]
				tot = 0
				id = curr_id
			else:
				if r[1]>max:
					max = r[1]
			tot += r[1]
		cids_retain[id] = round(float(max)/float(tot),2)
		prod_rsets[prod] = cids_retain
	
	test_data = []
	
	for r in rset:
		cid = r[0]
		#car value
		car_val = r[3]
		if car_val == "a":
			car_val = 1
		elif car_val == "b":
			car_val = 2
		elif car_val == "c":
			car_val = 3
		elif car_val == "d":
			car_val = 4
		elif car_val == "e":
			car_val = 5
		elif car_val == "f":
			car_val = 6
		elif car_val == "g":
			car_val = 7
		elif car_val == "h":
			car_val = 8
		elif car_val == "i":
			car_val = 9
		else:
			car_val = 0
		
			
		#risk factor
		risk_fac = r[4]
		if risk_fac == 1:
			risk_fac = 1
		elif risk_fac == 2: 
			risk_fac = 0.5
		elif risk_fac == 3: 
			risk_fac = 0.25
		elif risk_fac == 4: 
			risk_fac = 0
				
		age_old = r[7]
		age_yg = r[8]
		age_diff = round(float(age_old-age_yg)/float(age_old),2)
			
		#c_prev
		c_prev = r[6]
		if c_prev == r[18]:
			c_prev_retain = 1
		else:
			c_prev_retain = 0
			
		test_data.append([cid,r[1],r[2],car_val,risk_fac,r[5],c_prev_retain,age_diff,prod_rsets["a"][cid],prod_rsets["b"][cid],prod_rsets["c"][cid],prod_rsets["d"][cid],prod_rsets["e"][cid],prod_rsets["f"][cid],prod_rsets["g"][cid],r[16],r[17],r[18],r[19],r[20],r[21],r[22],r[23],r[9],r[10],r[11],r[12],r[13],r[14],r[15]])
			
	test_data = pd.DataFrame(test_data,columns=["cid","grp_size","homeowner","car_val","risk_factor","married","c_prev","age_diff","a_rt","b_rt","c_rt","d_rt","e_rt","f_rt","g_rt","avg_a","avg_b","avg_c","avg_d","avg_e","avg_f","avg_g","cost","a","b","c","d","e","f","g"])
	
	#print test_data
	
	a_sets = convert_df(test_data,"a")
	b_sets = convert_df(test_data,"b")
	c_sets = convert_df(test_data,"c")
	d_sets = convert_df(test_data,"d")
	e_sets = convert_df(test_data,"e")
	f_sets = convert_df(test_data,"f")
	g_sets = convert_df(test_data,"g")
	
	ct = 0
	for r in test_data.iterrows():
		
		cid = r[1]["cid"]
		print "customer--->"+str(cid)
		pred_a = models["a"]["classifier"].predict(models["a"]["feat_sel"].transform(a_sets[ct]))
		pred_b = models["b"]["classifier"].predict(models["b"]["feat_sel"].transform(b_sets[ct]))
		pred_c = models["c"]["classifier"].predict(models["c"]["feat_sel"].transform(c_sets[ct]))
		pred_d = models["d"]["classifier"].predict(models["d"]["feat_sel"].transform(d_sets[ct]))
		pred_e = models["e"]["classifier"].predict(models["e"]["feat_sel"].transform(e_sets[ct]))
		pred_f = models["f"]["classifier"].predict(models["f"]["feat_sel"].transform(f_sets[ct]))
		pred_g = models["g"]["classifier"].predict(models["g"]["feat_sel"].transform(g_sets[ct]))
		
		ct = ct+1
		
		opt_vectors = str(pred_a[0])+str(pred_b[0])+str(pred_c[0])+str(pred_d[0])+str(pred_e[0])+str(pred_f[0])+str(pred_g[0])
		
		feat_gen.write(str(cid)+","+opt_vectors+"\n")
	feat_gen.close()


def prepare_neural_models():
	train_df = pd.read_csv("train_set.csv")
	prod_options = ['a','b','c','d','e','f','g']
	
	neural_models = []
	for opt in prod_options:
		if opt == "a":
			train_cols =  [train_df.columns[3],train_df.columns[4],train_df.columns[15]]
		elif opt == "b":
			train_cols =  [train_df.columns[3],train_df.columns[4],train_df.columns[16]]
		elif opt == "c":
			train_cols =  [train_df.columns[3],train_df.columns[4],train_df.columns[17]]
		elif opt == "d":
			train_cols =  [train_df.columns[3],train_df.columns[4],train_df.columns[18]]
		elif opt == "e":
			train_cols =  [train_df.columns[3],train_df.columns[4],train_df.columns[19]]
		elif opt == "f":
			train_cols =  [train_df.columns[3],train_df.columns[4],train_df.columns[20]]
		elif opt == "g":
			train_cols =  [train_df.columns[3],train_df.columns[4],train_df.columns[21]]
			
		dataset = SupervisedDataSet(3,1)
		for df in train_df:
			dataset.addSample((df[1][train_cols[0]],df[1][train_cols[1]],df[1][train_cols[2]]),(df[opt],))
		#neural_ds.append(dataset)
	
		net = buildNetwork(3, 3, 1, bias=True, hiddenclass=TanhLayer)
		neural_trainer = BackpropTrainer(net,dataset)
		neural_trainer.train()
		neural_models.append(neural_trainer)	
		
	return neural_models
		
#prepare_trainset()
#cross_validate_model(10)
models = train_classifier()
predict_test(models)
'''

'''