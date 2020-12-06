class tenDParameters:
    def __init__(self,
    b: float,
    DM: float,
    pm_l: float,
    pm_b: float,
    vrad: float,
    sb: float,
    spml: float,
    spmb: float,
    sdm: float,
    vc: float):
        self.b = b
        self.DM = DM
        # this is actually pm_l * cos b, apparently
        self.pm_l = pm_l
        self.pm_b = pm_b
        self.vrad = vrad
        self.sb = sb
        self.spml = spml
        self.spmb = spmb
        self.sdm = sdm
        self.vc = vc