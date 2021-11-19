import numpy as np
from multiprocessing import Pool
import time
import torch
import argparse
from tqdm import tqdm
from vast.opensetAlgos.extreme_value_machine import ExtremeValueMachine

parser = argparse.ArgumentParser()
parser.add_argument("--train-feats", help="Features for training data")
parser.add_argument("--test-feats", help="Features for testing data")
parser.add_argument("--test-size", default=1024, help="Size of test")
parser.add_argument("--batch-size", default=100, help="Batch size for test")
parser.add_argument("--num-evaluation", default=10 ** 6, help="Number of evaluation")
parser.add_argument("--num-cpu", default=4, help="Number of cpu")
parser.add_argument("--max-percentage-early", default=5.0, help="Number of early max")
parser.add_argument("--evm-model", help="path to evm models")
parser.add_argument("--evm-batch-size", default=10000, help="Batch size for test", type=int)
parser.add_argument("--num-known-classes", default=29, help="Number of known classes")
parser.add_argument("--distance-function", default="cosine", help="Distance function")
parser.add_argument("--gpu-idx", default=0, help="gpu idx")

args = parser.parse_args()
TA1_test_size = args.test_size
batch_size = args.batch_size
number_of_evaluation = args.num_evaluation
n_cpu = args.num_cpu
max_percentage_of_early = args.max_percentage_early
train_dict = torch.load(args.train_feats)
test_dict = torch.load(args.test_feats)

t0 = time.time()

ond_train = torch.cat(train_dict["feats"])
ond_val = test_dict["known"]
ond_unknown = test_dict["unknown"]
evm_instance = ExtremeValueMachine.load(args.evm_model,
                                        device=f"cuda:{args.gpu_idx}")


ond_train = ond_train[~torch.any(ond_train.isnan(), dim=1)]
ond_val = ond_val[~torch.any(ond_val.isnan(), dim=1)]
ond_unknown = ond_unknown[~torch.any(ond_unknown.isnan(), dim=1)]



#L_train =  ond_train[:,0]
#sm_train =  ond_train[:,2]
#_val =  ond_val[:,0]
#sm_val =  ond_val[:,2]
#_unknown =  ond_unknown[:,0]
#sm_unknown =  ond_unknown[:,2]

p_train = []
for i in tqdm(range(0, ond_train.shape[0], args.evm_batch_size)):
    t1 = evm_instance.known_probs(ond_train[i:i+args.evm_batch_size].double())
    p_train.append(t1)
p_train = torch.cat(p_train).detach().cpu().numpy()
p_val =  evm_instance.known_probs(ond_val.double())
p_unknown =  evm_instance.known_probs(ond_unknown.double())
p_val = p_val.detach().cpu().numpy()
p_unknown = p_unknown.detach().cpu().numpy()
t1 = time.time()

def KL_Gaussian(mu, sigma, m, s):
  kl = np.log(s/sigma) + ( ( (sigma**2) + ( (mu-m) **2) ) / ( 2 * (s**2) ) ) - 0.5
  return kl


#mu_sm_train = np.mean(sm_train)
#sigma_sm_train = np.std(sm_train)
#mu_sm_val = np.mean(sm_val)
#sigma_sm_val = np.std(sm_val)
#mu_sm_unknown = np.mean(sm_unknown)
#sigma_sm_unknown = np.std(sm_unknown)
mu_p_train = np.mean(p_train)
sigma_p_train = np.std(p_train)
mu_p_val = np.mean(p_val)
sigma_p_val = np.std(p_val)
mu_p_unknown = np.mean(p_unknown)
sigma_p_unknown = np.std(p_unknown)


print("\nstart")
sigma_one_p_train = np.sqrt(np.mean((p_train-1.0)**2))


N_val = p_val.shape[0]

def task(n):
  rng = np.random.default_rng(n)
  ind = rng.choice(N_val, size=batch_size, replace=False)
  p_batch = p_val[ind]
  mu_p_batch = np.mean(p_batch)
  sigma_p_batch = np.std(p_batch)
  return KL_Gaussian(mu = mu_p_batch, sigma=sigma_p_batch, m = 1.0, s = sigma_one_p_train)


average_of_known_batch = int(TA1_test_size / (batch_size * 2) )

KL = np.zeros((number_of_evaluation, average_of_known_batch))

with Pool(n_cpu) as p:
  for j in range(average_of_known_batch):
    arr = (number_of_evaluation * j) + np.arange(number_of_evaluation)
    KL[:,j] = p.map(task, arr )

KL_evals = np.amax(KL, axis = 1)

KL_sorted = np.sort(KL_evals , kind = 'stable')

min_percentage_not_early = 100.0 - max_percentage_of_early
index = int(number_of_evaluation * (min_percentage_not_early/ 100 )) + 1
if index >= number_of_evaluation:
  index = - 1


t2 = time.time()

# print("\nJust Information SoftMax:")
# print("SoftMax: mu train = ", mu_sm_train)
# print("SoftMax: sigma train = ", sigma_sm_train)
# print("SoftMax: mu validation = ", mu_sm_val)
# print("SoftMax: sigma validation = ", sigma_sm_val)
# print("SoftMax: mu unknown = ", mu_sm_unknown)
# print("SoftMax: sigma unknown = ", sigma_sm_unknown)
# print("SoftMax: KL validation = ", KL_Gaussian(mu = mu_sm_val, sigma=sigma_sm_val, m = mu_sm_train, s = sigma_sm_train))
# print("SoftMax: KL unknown = ", KL_Gaussian(mu = mu_sm_unknown, sigma=sigma_sm_unknown, m = mu_sm_train, s = sigma_sm_train))

print("\nJust Information EVM:")
print("EVM: mu train = ", mu_p_train)
print("EVM: sigma train = ", sigma_p_train)
print("EVM: mu validation = ", mu_p_val)
print("EVM: sigma validation = ", sigma_p_val)
print("EVM: mu unknown = ", mu_p_unknown)
print("EVM: sigma unknown = ", sigma_p_unknown)
print("EVM: KL validation = ", KL_Gaussian(mu = mu_p_val, sigma=sigma_p_val, m = mu_p_train, s = sigma_p_train))
print("EVM: KL unknown = ", KL_Gaussian(mu = mu_p_unknown, sigma=sigma_p_unknown, m = mu_p_train, s = sigma_p_train))



print("\n\nINFO: assumed mu train = ", 1.0)
print("computed sigma train = ", sigma_one_p_train)
print("computed KL validation = ", KL_Gaussian(mu = mu_p_val, sigma=sigma_p_val, m = 1.0, s = sigma_one_p_train))
print("computed unknown validation = ", KL_Gaussian(mu = mu_p_unknown, sigma=sigma_p_unknown, m = 1.0, s = sigma_one_p_train))
print("optimized threshold for KL_validation is ", KL_sorted[index])


print("\n\nJust Information:")
print("Minimum KL of validation trials is ", KL_sorted[0])
print("First quartile KL of validation trials is ", KL_sorted[int(number_of_evaluation* 0.25)])
print("Median KL of validation trials is ", KL_sorted[int(number_of_evaluation * 0.5)])
print("Third quartile KL of validation trials is ", KL_sorted[int(number_of_evaluation* 0.75)])
print("Maximum KL of validation trials is ", KL_sorted[-1])

print("\n\nSummary for Action:")
print("mu train = ", 1.0)
print("threshold for KL_validation = ", KL_sorted[index])

print("\nLoading time = ", t1-t0)
print("Processing time = ", t2-t1)
print("Total time = ", t2-t0)
print("End\n")
