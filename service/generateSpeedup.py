from itertools import combinations
apps=('pos', 'ner', 'imc', 'dig', 'face')
low_apps=('pos', 'ner')
med_apps=('dig') # not really used! just listed to show the level of speedup
high_apps=('imc','face')
c_lists = list(combinations(apps,4))

test_cases = []
for case1 in c_lists:
	# pick 2 to run on cpu from case1
	cpu_list=list(combinations(case1,2))
	for case2 in cpu_list:
		temp_case = case2
		for i in range(len(case1)):
			if case1[i] not in case2:
				temp_case = temp_case + (case1[i],)
		test_cases.append(temp_case)
cnt=1
for tcase in test_cases:
	file_name="speedup"+str(cnt)+".txt"
	with open(file_name,"w") as fp:
	  	fp.write(tcase[0]+","+"0.5"+"\n")
  		fp.write(tcase[1]+","+"0.5"+"\n")
  		fp.write(tcase[2]+","+"1.5"+"\n")
  		fp.write(tcase[3]+","+"1.5"+"\n")
	cnt = cnt + 1 
