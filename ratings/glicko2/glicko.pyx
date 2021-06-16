import math

cdef class GlickoRating:
    cdef public double mu, phi, sigma
    cdef readonly double elo_scale
    cdef readonly double glicko_scale
    
    def __cinit__(self, double rating, double rd, double sigma):
        self.elo_scale = 1500.
        self.glicko_scale = 173.7178
        self.mu = (rating - self.elo_scale) / self.glicko_scale
        self.phi = rd / self.glicko_scale
        self.sigma = sigma
        
    cdef double get_rating(self):
        return self.mu * self.glicko_scale + self.elo_scale

    cdef double get_rd(self):
        return self.phi * self.glicko_scale        
        

cdef class Glicko2:
    
    cdef double tau
    cdef dict rating_dict
    def __init__(self, double tau):
        self.tau = tau
        self.rating_dict = {}

    def __getitem__(self, player: str) -> float:
        return self.rating_dict[player]

    def __setitem__(self, fighter: str, fighter_rating: GlickoRating) -> None:
        self.rating_dict[fighter] = fighter_rating

    def add_player(self, name: str, rating: float = 1500., ):
        self.rating_dict[name] = rating
        
    cdef double get_g_factor(self, double phi):
        return 1. / math.sqrt(1. + 3. * phi**2 / math.pi**2)
    
    cdef double get_expect_result(self, double mu_p1, double mu_p2, double g_p2):
        return 1. / (1 + math.exp(-g_p2 * (mu_p1 - mu_p2)))
    
    cdef double update_volatility(self, double vol, double phi, double Delta, double v):
        
        def check_fun(double x):
            
            cdef double term_a = math.exp(x) * (Delta**2 - phi**2 -v - math.exp(x))
            cdef double term_b = 2 * (phi**2 + v + math.exp(x))**2
            cdef double term_c = x - a
            cdef double term_d = self.tau**2
            
            return term_a / term_b - term_c / term_d
        
        cdef double tol = 1e-6
        cdef double a = math.log(vol**2)
        
        cdef double A = math.log(vol**2)
        cdef double B, k, b_check
        
        if Delta**2 > phi**2 + v:
            B = math.log(Delta**2 - phi**2 - v)
        else:
            k = 1.
            b_check = check_fun(a - k * self.tau)
            
            while b_check < 0:
                k += 1
                b_check = check_fun(a - k * self.tau)
            B = a - k * self.tau
            
        cdef double fA = check_fun(A)
        cdef double fB = check_fun(B)
        cdef double C, fC
        
        while abs(B - A) > tol:
            C = A + (A - B) * ( fA / (fB - fA))
            fC = check_fun(C)
            
            if fC * fB < 0.:
                A, fA = B, fB
            else:
                fA = fA / 2.
                
            B, fB = C, fC
            
        cdef double new_vol = math.exp(A / 2.)
        
        return new_vol
        
    def update_ratings(self, p1_name: str, p2_name: str, score: float) -> None:
        
        # Step 2
        cdef GlickoRating p1 = self.rating_dict[p1_name]
        cdef GlickoRating p2 = self.rating_dict[p2_name]
        
        # Step 3
        cdef double p1_g = self.get_g_factor(p1.phi)
        cdef double p2_g = self.get_g_factor(p2.phi)
        
        cdef double p1_e = self.get_expect_result(p1.mu, p2.mu, p2_g)
        cdef double p2_e = self.get_expect_result(p2.mu, p1.mu, p1_g)
        
        cdef double p1_v = 1. / (p2_g**2 * p1_e * (1 - p1_e))
        cdef double p2_v = 1. / (p1_g**2 * p2_e * (1 - p2_e))
        
        # Step 4
        cdef double p1_Delta = p1_v * p2_g * (score - p1_e)
        cdef double p2_Delta = p2_v * p1_g * (score - p2_e)
        
        # Step 5
        cdef double p1_updated_sigma = self.update_volatility(p1.sigma, p1.phi, p1_Delta, p1_v)
        cdef double p2_updated_sigma = self.update_volatility(p2.sigma, p2.phi, p2_Delta, p2_v)
        
        # Step 6
        cdef double p1_phi_star = math.sqrt(p1.phi**2 + p1_updated_sigma**2)
        cdef double p2_phi_star = math.sqrt(p2.phi**2 + p2_updated_sigma**2)
        
        # Step 7
        cdef double p1_updated_phi = 1. / math.sqrt((1. / p1_phi_star**2) + (1. / p1_updated_sigma**2))
        cdef double p2_updated_phi = 1. / math.sqrt((1. / p2_phi_star**2) + (1. / p2_updated_sigma**2))
        
        cdef double p1_updated_mu = p1.mu + p1_updated_phi**2 * p2_g * (score - p1_e)
        cdef double p2_updated_mu = p2.mu + p2_updated_phi**2 * p1_g * (score - p2_e)
        
        p1.mu = p1_updated_mu
        p1.phi = p1_updated_phi
        p1.sigma = p1_updated_sigma
        
        p2.mu = p2_updated_mu
        p2.phi = p2_updated_phi
        p2.sigma = p2_updated_sigma
        
