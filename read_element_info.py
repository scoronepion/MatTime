import pandas as pd

def get_info():
    '''
        Eea - Electron affinity - 电子亲和性
        I1 - First ionization potential - 第一电离电位
        I2 - Second ionization potential - 第二电离电位
        Tm -  Melting point - 熔点
        AW - Atomic weight - 原子量
        AN - Atomic number - 原子序数
        Rm - Metallic radius - 金属半径
        Rc - Covalent radius - 共价半径
        Gp - Group - 群
        P - Period - 周期
        VEC - Valence electrons - 电子化合价
        sVEC
        pVEC
        dVEC
        XP - Pauling electronegativity - 泡林电负性
        XM - Mulliken electronegativity - 密立根电负度
        Cp - Heat capacity - 热容
        K - Thermal conductivity - 导热系数
        W - Work function - 逸出功 / 自由能
        D -  Density - 密度
        Hf - Heat of fusion - 熔化热
        LP - Lattice volume - 晶格体积
        Tb - Boiling point - 沸点
    '''
    raw = pd.read_csv('element_info.csv')
    print(raw.info())
    print(raw.head())

if __name__ == '__main__':
    get_info()