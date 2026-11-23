import numpy as np
from scipy.special import iv
from scipy.stats import wrapcauchy
from scipy.stats import vonmises
import csv

class Distribution:
    def pdf(self):
        pass

    def sample(self):
        pass

    def likelihood(self, x):
        acc = np.prod(self.pdf(x))
        return acc            

    def log_likelihood(self, x):
        acc = np.sum(np.log(self.pdf(x)))
        return acc


class Von_mises(Distribution) : 
    def __init__(self, mean, kappa):
        self.mean = mean

        if kappa <= 0:
            raise ValueError("kappa must be in (0, +inf)")
 
        self.kappa = kappa    
    
    def pdf(self, x):
        return vonmises.pdf(x, self.kappa, loc=self.mean)
    
    def cdf(self, x) :   
        return vonmises.cdf(x, self.kappa, loc=self.mean)

    def ppf(self, q) :
        x = vonmises.ppf(q, self.kappa, loc=self.mean)
        # Map to [-pi, pi]
        x = np.where(x > np.pi, x - 2 * np.pi, x)
        return x
     
    def sample(self, size=1):
        return vonmises.rvs(self.kappa, loc=self.mean, size=size)


class Wrapped_cauchy(Distribution) :
    def __init__(self, mean, rho):
        self.mean = mean
        
        if rho <= 0 or rho >= 1:
            raise ValueError("rho must be in (0, 1)")
        
        self.rho = rho
    
    def pdf(self, x):
        x_wrapped = self.mean + np.mod(x - self.mean, 2*np.pi)
        return wrapcauchy.pdf(x_wrapped, self.rho, loc=self.mean)

    def cdf(self, x) :
        x_wrapped = self.mean + np.mod(x - self.mean, 2*np.pi)
        return wrapcauchy.cdf(x_wrapped, self.rho, loc=self.mean)

    def ppf(self, q):
        x = wrapcauchy.ppf(q, self.rho, loc=self.mean)
        # Map to [-pi, pi]
        x = np.where(x > np.pi, x - 2 * np.pi, x)
        return x

    def sample(self, size=1):
        return wrapcauchy.rvs(self.rho, loc=self.mean, size=size)


class Joint_von_mises(Distribution) :
    def __init__(self, means, kappas):

        if any(k < 0 for k in kappas):
            raise ValueError("kappas must be in (0, +inf)")
        self.marginal_distributions = [Von_mises(mean, kappa) for mean, kappa in zip(means, kappas)]

        self.d = len(self.marginal_distributions)
        
    def pdf(self, x):
        marginals_density = np.ones(x.shape[0])

        for i in range(self.d) :
            marginals_density *= self.marginal_distributions[i].pdf(x[:,i])
        
        return marginals_density
 
    def sample(self, size=1):        
        sample = []
        for i in range(self.d):
            marg_sample = self.marginal_distributions[i].sample(size)
            # map to [-pi,pi]
            marg_sample = ((marg_sample + np.pi) % (2 * np.pi)) - np.pi
            sample.append(marg_sample)

        return np.column_stack(sample)


class Von_mises_CBMD(Distribution) :
    def __init__(self, marginal_distributions, qs, kappas):
        self.marginal_distributions = marginal_distributions

        if any(q not in (1, -1) for q in qs):
            raise ValueError("qs must be either 1 or -1")
        self.qs = qs
  
        if any(k < 0 for k in kappas):
            raise ValueError("kappas must be in (0, +inf)")
        self.kappas = kappas

        self.d = len(self.marginal_distributions)
        
    def pdf(self, x):
        
        circular_uniform_x = []

        for i in range(self.d):
            circular_uniform_x.append(2 * np.pi * self.marginal_distributions[i].cdf(x[:,i]))

        circular_uniform_x = np.column_stack(circular_uniform_x)

        cos_accs = np.zeros(x.shape[0])
        sin_accs = np.zeros(x.shape[0])
        bessel_norm = 1.0

        for i in range(self.d) :
          cos_accs += self.kappas[i] * np.cos(circular_uniform_x[:,i])
          sin_accs += self.kappas[i] * self.qs[i] * np.sin(circular_uniform_x[:,i])
          bessel_norm *= iv(0, self.kappas[i])
        

        Rs = np.sqrt(np.power(cos_accs, 2) + np.power(sin_accs, 2))
        numerators = iv(0, Rs)
        denominator = np.power(2 * np.pi, self.d) * bessel_norm

        circula_density = numerators / denominator

        marginals_density = np.ones(x.shape[0])

        for i in range(self.d) :
            marginals_density *= self.marginal_distributions[i].pdf(x[:,i])
        
        return pow(2.0 * np.pi, self.d) * marginals_density * circula_density
 
    def sample(self, size=1):
        phi = np.random.uniform(size=size) * 2 * np.pi

        circula_shifts = []
        for i in range(self.d):
            circula_shifts.append(vonmises.rvs(self.kappas[i], loc=0, size=size))
        circula_shifts = np.column_stack(circula_shifts)


        uniforms = []
        for i in range(self.d):
            shifted = (circula_shifts[:,i] + self.qs[i] * phi) % (2 * np.pi)
            uniforms.append(shifted / (2 * np.pi))
        uniforms = np.column_stack(uniforms)

        sample = []
        for i in range(self.d):
            marg_sample = self.marginal_distributions[i].ppf(uniforms[:,i])
            # map to [-pi,pi]
            marg_sample = ((marg_sample + np.pi) % (2 * np.pi)) - np.pi
            sample.append(marg_sample)

        return np.column_stack(sample)




class Wrapped_cauchy_CBMD(Distribution) :
    def __init__(self, marginal_distributions, qs, rhos):
        self.marginal_distributions = marginal_distributions

        if any(q not in (1, -1) for q in qs):
            raise ValueError("qs must be either 1 or -1")
        self.qs = qs
  
        if any(rho <= 0 or rho >= 1 for rho in rhos):
            raise ValueError("rhos must be in (0, 1)")
        self.rhos = rhos 
        
        self.d = len(self.marginal_distributions)

    def stabilize_rhos(self):
        #rhos cannot be the exact same values, perturb them a tiny bit
        for i in range(self.d):
            for j in range(i + 1, self.d):
                if self.rhos[i] == self.rhos[j]:
                    self.rhos[j] += 1e-5

    def pdf(self, x):
        self.stabilize_rhos()

        circular_uniform_x = []

        for i in range(self.d):
            circular_uniform_x.append(2 * np.pi * self.marginal_distributions[i].cdf(x[:,i]))

        circular_uniform_x = np.column_stack(circular_uniform_x)
        
        circula_density = np.zeros(x.shape[0], dtype=complex)

        for i in range(self.d):
            circula_density_part = np.power(self.rhos[i] * np.exp(1j * self.qs[i] * circular_uniform_x[:,i]), self.d - 1)

            for j in range(self.d):
                if i == j:
                    continue
                    
                num = (1 - self.rhos[j]**2)
                denom_left = (self.rhos[i] * np.exp(1j * self.qs[i] * circular_uniform_x[:,i]) - self.rhos[j] * np.exp(1j * self.qs[j] * circular_uniform_x[:,j]))
                denom_right = (1 - self.rhos[i] * self.rhos[j] * np.exp(1j * (self.qs[i] * circular_uniform_x[:,i] - self.qs[j] * circular_uniform_x[:,j])))

                prodpart = num / (denom_left * denom_right)
                
                circula_density_part *= prodpart
                
            circula_density += circula_density_part
        
        circula_density /= np.power(2 * np.pi, self.d)
        circula_density = np.real(circula_density)

        marginals_density = np.ones(x.shape[0])

        for i in range(self.d) :
            marginals_density *= self.marginal_distributions[i].pdf(x[:,i])
        
        return pow(2.0 * np.pi, self.d) * marginals_density * circula_density
    
    def sample(self, size=1):
        phi = np.random.uniform(size=size) * 2 * np.pi

        circula_shifts = []
        for i in range(self.d):
            circula_shifts.append(wrapcauchy.rvs(self.rhos[i], loc=0, size=size))
        circula_shifts = np.column_stack(circula_shifts)


        uniforms = []
        for i in range(self.d):
            shifted = (circula_shifts[:,i] + self.qs[i] * phi) % (2 * np.pi)
            uniforms.append(shifted / (2 * np.pi))
        uniforms = np.column_stack(uniforms)

        sample = []
        for i in range(self.d):
            marg_sample = self.marginal_distributions[i].ppf(uniforms[:,i])
            
            # map to [-pi,pi]
            marg_sample = ((marg_sample + np.pi) % (2 * np.pi)) - np.pi
            sample.append(marg_sample)

        return np.column_stack(sample)



class Mixture_model(Distribution) :
    def __init__(self, components, weights):
        self.components = components
       
        total = sum(weights)
        normalized = [w / total for w in weights]

        self.weights = normalized
        self.d = components[0].d
      
        self.k = len(components)

    def pdf(self, x):
        acc = np.zeros(x.shape[0])

        for i in range(self.k):
            acc += self.weights[i] * self.components[i].pdf(x)
    
        return acc

    def sample(self, size=1):
        selected_component = np.random.choice(self.k, p=self.weights, size=size)
        components_sample_sizes = np.bincount(selected_component, minlength=self.k)


        sample = []

        for i in range(self.k):
            if (components_sample_sizes[i] == 0):
                continue
            sample.append(self.components[i].sample(size=components_sample_sizes[i]))
        
        sample = np.vstack(sample)
        np.random.shuffle(sample)
        return sample

    @classmethod
    def from_file(self, filename, component_type):

        if component_type not in {"baseline", "VMVM", "VMWC"}:
            raise ValueError("Component distribution type must be 'baseline', 'VMVM' or 'VMWC'")

        weights = []
        components = []
        d = 0

        with open(filename, mode='r') as f:
            csv_file = csv.reader(f)
            title = False
            for line in csv_file:
                if not title:
                    title=True
                    continue
            
                weights.append(float(line[0]))

                if component_type == "baseline":
                    d = round((len(line) - 1) / 2)
                    means = []
                    kappas = []               
                    
                    for i in range(d):
                        means.append(float(line[1 + (i * 2)]))
                        kappas.append(float(line[1 + (i * 2) + 1]))
                    components.append(Joint_von_mises(means, kappas))
                
                else: 
                    d = round((len(line) - 1) / 4)
                    marginal_distributions = []
                    circula_concentrations = []
                    qs = []
                
                    for i in range(d):
                        mean = float(line[1 + (i * 4)])
                        concentration = float(line[1 + (i * 4) + 1])
                        concentration = 1e-5 if concentration == 0 else concentration
                        marginal_distributions.append(Von_mises(mean, concentration))
                        
                        circula_concentration = float(line[1 + (i * 4) + 2])
                        circula_concentration = 1e-5 if circula_concentration == 0 else circula_concentration
                        circula_concentrations.append(circula_concentration)

                        qs.append(float(line[1 + (i * 4) + 3]))
                    
                    if component_type == "VMVM":
                        components.append(Von_mises_CBMD(marginal_distributions, qs, circula_concentrations))
                    else:
                        components.append(Wrapped_cauchy_CBMD(marginal_distributions, qs, circula_concentrations))

        return Mixture_model(components, weights)



    @classmethod
    def to_file(self, filename, mixture_model, component_type):
 
        if component_type not in {"baseline", "VMVM", "VMWC"}:
            raise ValueError("Component distribution type must be 'baseline', 'VMVM' or 'VMWC'")

        with open(filename, mode="w", newline="") as f:
            csv_file = csv.writer(f)

            # CSV header
            if component_type == "baseline":
                header = ["weight"]
                d = len(mixture_model.components[0].means)

                for i in range(d):
                    header.extend([f"mean{i}", f"kappa{i}"])

            else:
                header = ["weight"]
                d = len(mixture_model.components[0].marginal_distributions)

                for i in range(d):
                    header.extend([f"mean{i}", f"marg_var{i}", f"rho{i}", f"q{i}"])

            csv_file.writerow(header)

            # Write each component
            for weight, component in zip(mixture_model.weights, mixture_model.components):
                row = [weight]

                if component_type == "baseline":
                    for mean, kappa in zip(component.means, component.kappas):
                        row.extend([mean, kappa])

                else:
                    for marginal, q, rho in zip(component.marginal_distributions, component.qs, component.rhos):
                        row.extend([marginal.mean, marginal.kappa, rho, q])

                csv_file.writerow(row)

