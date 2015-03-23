from MCMC_Model import *

def initialize_mcmc(model):
	"""This is just to initialize the pickle db to store the MCMC state"""
	mcmc = pm.MCMC(model, name="MCMC", db="pickle", dbname="sg_mcmc.pickle")
	mcmc.db
	mcmc.sample(iter=10, burn=1, thin=1)
	mcmc.db.close()
	
	return model
	
def sample_mcmc(model, sampling_size):
	db = pm.database.pickle.load('sg_mcmc.pickle')
	#print "x0 trace: ",len(db.trace('x0',chain=None)[:])
	#print "x1 trace: ",len(db.trace('x1',chain=None)[:])
	mcmc = pm.MCMC(model, name="MCMC", db=db)
	mcmc.sample(iter=sampling_size, burn=10, thin=2)
	mcmc.db.close()
	
	return model
	
	
