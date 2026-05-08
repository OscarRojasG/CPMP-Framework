from generation.instances.instance_generator import InstanceGenerator
from generation.instances import RandomMovesGenerator, FullRandomGenerator
import numpy as np
from scipy.stats import norm
from scipy.optimize import nnls
from utils.utils import distribuir_suma_exacta
    
class UniformCostGenerator(InstanceGenerator):
    def __init__(self, S, seed):
        super().__init__(H=S+2, S=S, N=S*S, seed=seed)

    def generate_instances(self, amount):
        n_opt, R = self.resolver_mezcla_distribuciones(self.S, amount)
        n_opt = distribuir_suma_exacta(n_opt, amount)

        for i, (r, n) in enumerate(zip(R, n_opt)):
            if i < len(R) - 1:
                gen = RandomMovesGenerator(self.H, self.S, self.N, r=round(r), seed=self.seed)
            else:
                gen = FullRandomGenerator(self.H, self.S, self.N, seed=self.seed)

            instances = gen.generate_instances(n)
            for instance in instances:
                self.add_instance(instance)
        
        return self.instances
    
    def resolver_mezcla_distribuciones(self, S, N):
        # 1. Límite teórico (Usa los coeficientes optimizados de tu regresión)
        A_teorico = 0.9791 * S**2.1686 
        K_max = int(np.floor(A_teorico - 1))
        
        # 2. Bines de destino
        pasos_k = np.arange(1, K_max + 1)
        
        # 3. Candidatos R con alta densidad en el eje de las medias (mu)
        recursos_R = [self.calcR(S, k) for k in range(S, K_max + 1)]
        m = len(recursos_R)
        
        # 4. Construcción de la Matriz P corregida
        P_full = np.zeros((m, K_max))
        for i, R_i in enumerate(recursos_R):
            mu_i, sigma_i = self.dist(S, R_i)
            
            # CORRECTO: Usar pasos_k para evaluar la densidad en cada bin
            cdf_upper = norm.cdf(pasos_k + 0.5, loc=mu_i, scale=sigma_i)
            cdf_lower = norm.cdf(pasos_k - 0.5, loc=mu_i, scale=sigma_i)
            
            fila = cdf_upper - cdf_lower
            P_full[i, :] = fila / (np.sum(fila) + 1e-12)

        # 5. Recorte para uniformidad (Ignoramos el inicio inestable)
        inicio_bin = S
        P_reducida = P_full[:, inicio_bin-1:]
        
        K_recortado = K_max - (inicio_bin - 1)
        F = N / K_max
        b_reducido = np.full(K_recortado, F)
        
        # 6. Optimización NNLS
        n_opt, _ = nnls(P_reducida.T, b_reducido)
        
        return n_opt, recursos_R

    def dist(self, S, R):
        mu = self.mu(S, R)
        sigma = self.sigma(S, R)
        return mu, sigma
    
    def sigma(self, S, R):
        # Coeficientes optimizados que obtuviste
        a0, a1, b0, b1, c0, c1, p = [0.36144427, -0.21628687, 0.6997611, 0.04425758, -0.12492023, 0.00627366, 1.03160386]

        if R <= 1.0: return 0.0001 # Evita sigma=0 absoluto en cálculos

        lx2 = np.log(R)
        A = a0 + a1 * S
        B = b0 + b1 * S
        C = c0 + c1 * S

        # Modelo anclado
        r_term = R - 1
        exponencial = np.exp(A + B * lx2 + C * (lx2**2))
        factor_anclaje = (r_term**p) / (1 + r_term**p)

        val = exponencial * factor_anclaje

        return val

    def mu(self, S, R):
        # Parámetros optimizados de la nueva regresión anclada
        L_s = 1.2582 * S**2.0648
        C_s = np.exp(0.6080 * S + 1.0533)
        p = 0.9469

        # Término R-1 para el anclaje
        r_term = np.maximum(R - 1, 0)

        # Modelo: 1 + (L_s - 1) * [ (R-1)^p / (C_s + (R-1)^p) ]
        factor_saturacion = (r_term**p) / (C_s + r_term**p)
        return 1 + (L_s - 1) * factor_saturacion

    def calcR(self, S, x_i):
        # Parámetros optimizados
        L_s = 1.2582 * S**2.0648
        C_s = np.exp(0.6080 * S + 1.0533)
        p = 0.9469

        # Ajustamos x_i para que no supere el techo asintótico L_s
        # y restamos 1 por el anclaje
        Y = np.maximum(x_i - 1, 1e-6)
        M = L_s - 1

        # Evitamos valores fuera de dominio (x_i no puede superar o igualar la asíntota L_s)
        if Y >= M:
            return float('inf')

        # Cálculo de la inversa
        return 1 + ((Y * C_s) / (M - Y))**(1 / p)