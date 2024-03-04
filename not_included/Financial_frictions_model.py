#Financial frictions model
class FirmEnvironment:
    def __init__(self, num_firms, num_periods):
        self.num_firms          = num_firms
        self.num_periods        = num_periods
        self.discount           = 0.956
        self.risk_free          = 0.04
        self.depriciation_rate  = 0.07
        self.alpha              = 0.7           #return to scale
        self.agg_prod           = 1
        self.pers_prod_mean     = -1.2591
        self.pers_prod_var      = 0.1498
        self.pers_prod_pers     = 0.9
        self.c                  = 0.7           #fixed cost
        self.sigma              = 0.3
        self.miu                = 0.25
        self.end_ent_upper      = 9.7
        self.networth         = np.random.uniform(0, 10, num_firms)
        self.productivity       = np.random.normal(0.5, 0.2, num_firms)
        self.productivity_shock = np.random.normal(0.5, 0.2, num_firms)

