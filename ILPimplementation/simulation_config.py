class SimulationConfig:
    def __init__(self, wharf_df, line_df, headway_df, tt_df, vessel_df, # DataFrames
                 initial_time, period_length, Tset, # Simulation Time Parameters
                 Lset, Bc, B, Bplus, Jset, Wset, Dset, Vset, Zset,  # Set Definitions
                 Dc, nc, Tc, # Crew pause Parameters, and fuctions .py file
                 rv_plus, pc, functions): # Charging Parameters, and fuctions .py file
        self.wharf_df = wharf_df
        self.line_df = line_df
        self.headway_df = headway_df
        self.tt_df = tt_df
        self.vessel_df = vessel_df
        self.initial_time = initial_time
        self.period_length = period_length
        self.Tset = Tset
        self.Lset = Lset
        self.Bc = Bc
        self.B = B
        self.Bplus = Bplus
        self.Jset = Jset
        self.Wset = Wset
        self.Dset = Dset
        self.Vset = Vset
        self.Zset = Zset
        self.Dc = Dc
        self.nc = nc
        self.Tc = Tc
        self.rv_plus = rv_plus
        self.pc = pc
        self.functions = functions



