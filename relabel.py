import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
output_fileother = open("train.txt","w")
for filename in os.listdir(os.getcwd()):
	if("Benign" in dir_path+filename ):
   	 		output_fileother.write(dir_path + "/"+filename + " " + str(0) + '\n')
   	if("Malign" in dir_path+filename): 		
   			output_fileother.write(dir_path +"/" +filename + " " + str(1) + '\n')
	
output_fileother.close()

