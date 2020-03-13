import subprocess
import time
import numpy as np

"""Creates an opt file based on a template, where the P wave amplitude and seed are modified."""
def CreateTmpFile(lines, linestoEdit, factor, new_seed):
    for line in linestoEdit:
        channel = lines[line].split()  # ge the line with the decay channel, split into list of letters
        amplitude = float(channel[2])*factor  # modify the amplitude of the P wave by factor
        channel[2] = str(amplitude)
        channel = ' '.join(channel)  # recombine the list into a string
        seed = lines[3].split()  # get the seed line
        seed[1] = str(new_seed)  # change the seed to the random value
        lines[3] = ' '.join(seed)
        lines[line] = channel  # replace the uneditted line with the editted one

    for line in lines:
        writefile.write(line + '\n')  # write the opt file

    writefile.close()  # close the opt file


"""Runs a ccommand line in a console."""
def run_cmd(cmd):
	process = subprocess.Popen(cmd.split(), stdout = subprocess.PIPE)  # opens a new subprocess, takes command line as input
	# use this to print the console line by line for debugging
    #for line in process.stdout:
	#	print line
	process.communicate()  # close the subprocess, no outputs required
	process.stdout.close()
	return " "


"""Main Body"""
linestoEdit = [5, 6, 7, 8]  # which line in the template file need to be modified
seeds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # numbering of the files, completely aribtrary
factors = [1E-5, 1E-4, 1E-3, 1E-2, 1E-1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 1E1, 1E2, 1E3, 1E4, 1E5]  # relative p wave amplitudes to generate

times = []
for j in range(len(seeds)):
	for i in range(len(factors)):
		seedToUse = np.random.randint(0, 100000)  # pick a random integer as the seed
		print "seed:" + str(seedToUse)  # print the seed
		print('working...')
		print(factors[i])  # which factor are we using
		file = open('AutomatedOptFile.opt', 'r+')  # open the template file
		lines = file.read().splitlines()  # split the document into a list o the lines
		writefile = open('tmp.opt', 'a')  # create the optfile with the modifications made to the template
		writefile.truncate(0)  # we want to replace this with a blank one if it already exists
		CreateTmpFile(lines, linestoEdit, factors[i], seedToUse)  # call function to modify the temp opt file, based of the template
		file.close()  # close the temporary file
		s = time.time()  # start timing
		run_cmd("./AmpGen/build/bin/Generator tmp.opt --nEvents=10000 --Output=Amp_B_" + str(i) + "_S" + str(seeds[j]) + ".root")  # run the command in the terminal, modify nEvents to change the nummber of events
		e = time.time()  # stop timing
		times.append(e-s)
		print 'time taken:' + str(e-s)  # print generation time


totalTime = 0
for i in range(len(times)):
	totalTime += times[i]

print "Total Time: " + str(totalTime)  # print total computation time

