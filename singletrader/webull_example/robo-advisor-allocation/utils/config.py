import pandas as pd


asset_df = pd.DataFrame([
    ('equity','Large Cap Growth Equities','S&P500 Index','IVV'),
    ('equity','Large Cap Blend Equities','Dow Jones U.S. Dividend 100 Index','SCHD'),
    ('fixed income', 'Total Bond Market','Bloomberg US Aggregate','AGG'),
    ('fixed income', 'Inflation-Protected Bonds','TIPS','SCHP')
],columns=['asset type','sub-asset type','Proxy Index','ETF'])


#asset class of wealthfront
asset_wealthfront = pd.DataFrame([
    ('equity','US stocks','CRSP US Total Market Index','VTI',.45),
    ('equity','Foreign Developed Stocks','MSCI EAFE Index','EFA',.35),
    ('equity','Emerging Market Stocks','MSCI Emerging Markets Index','IEMG',.35),
    ('equity','Dividend Stocks','Dow Jones U.S. Dividend 100 Index','SCHD',.35),
    ('fixed income', 'US Govt Bonds','Bloomberg US Aggregate','AGG',.35),
    ('fixed income', 'US Corporate Bonds','iBoxx USD Liquid Investment Grade Index','LQD',.35),
    ('fixed income', 'Emerging Market Bonds','J.P. Morgan EMBI Global Core Index','EMB',.0),
    ('fixed income', 'Municipal Bonds ','S&P National AMT-Free Municipal Bond','VTEB',.35),
    ('fixed income', 'TIPS','Barclays US Inflation-linked Bond Index','SCHP',.35),
    ('inflation asset', 'real estate','Dow Jones U.S. Real Estate Index','IYR',0),
    ('inflation asset', 'Commodities','Energy Select Sector Index','XLE',.35),
],columns=['asset type','sub-asset type','Proxy Index','ETF','max_weight'])


#asset class of webull
asset_webull = pd.DataFrame([
    ('equity','US large cap','S&P500 Index','SPY',.3,'IVV'),
    ('equity','US mid cap','S&P 400® Mid Cap','SPMD',.3,'IWR'),
    ('equity','US small cap','S&P 600® Smal Cap','SPSM',0.3,'IJR'),
    ('equity','Foreign Developed Stocks','MSCI EAFE Index','EFA',.3,'SPDW'),
    ('equity','Emerging Market Stocks','MSCI Emerging Markets Index','IEMG',.3,'EEM'),
    ('equity','Dividend Stocks','Dow Jones U.S. Dividend 100 Index','SCHD',.3,'VYM'),
    ('fixed income', 'US Govt Bonds','Bloomberg US Aggregate','AGG',.3,'BND'),
    ('fixed income', 'US Corporate Bonds','iBoxx USD Liquid Investment Grade Index','LQD',.3,'VCIT'),
    ('fixed income', 'Emerging Market Bonds','J.P. Morgan EMBI Global Core Index','EMB',.0,'VWOB'),
    ('fixed income', 'Municipal Bonds ','S&P National AMT-Free Municipal Bond','SPTI',.3,'SHY'),
    ('fixed income', 'TIPS','Barclays US Inflation-linked Bond Index','SCHP',.3,'TIP'),
    ('inflation asset', 'real estate','Dow Jones U.S. Real Estate Index','IYR',0,'VNQ'),
    ('inflation asset', 'Commodities','Energy Select Sector Index','XLE',.1,'VDE'),
],columns=['asset type','sub-asset type','Proxy Index','ETF','max_weight','alternative'])


#asset class of statestreet
asset_statestreet = pd.DataFrame([
    ('equity','US large cap','S&P500 Index','SPY',1),
    ('equity','US sector rotation','PDR® SSGA US Sector Rotation ETF','XLSR',1),
    ('equity','US mid cap','S&P 400® Mid Cap','SPMD',1),
    ('equity','US small cap','S&P 600® Smal Cap','SPSM',1),
    ('equity','developed equity','SPDR® Portfolio Developed World ex-US','SPDW',1),
    ('equity','international small cap','SPDR® S&P® International Smal Cap','GWX',1),
    ('equity','europe equity','SPDR® Portfolio Europe ETF','SPEU',1),
    ('equity','emerging equity','SPDR® Portfolio Emerging Markets','SPEM',1),
    ('equity','Pacific equity','Vanguard FTSE Pacific ETF','VPL',1),
    ('fixed income','US bonds','SPDR® Portfolio Aggregate Bond ETF','SPAB',1),
    ('fixed income','US bonds','SPDR® SSGA Fixed Income Sector Rotation','FISR',1),
    ('fixed income','TIPS','SPDR® Bloomberg 1-10 Year TIPS ETF','TIPX',1),
    ('fixed income','Municipal Bonds','SPDR® Portfolio Long Term Treasury ETF','SPTL',1),
    ('fixed income','Coporate Bonds','SPDR® Blackstone Senior Loan ETF','SRLN',1),
    ('fixed income','emerging Bonds','SPDR® Bloomberg Emerging Markets Local Bond ETF','EBND',1),
    ('fixed income','Municipal Bonds','SPDR® Portfolio Intermediate Term Treasury ETF','SPTI',1),
    ('fixed income','US bonds','SPDR® Bloomberg High Yield BondETF','JNK',1),
    ('fixed income','Municipal Bonds','PDR® Bloomberg 1-3 Month T-Bill ETF','BIL',1),
    ('inflation asset','commodities','Invesco Optimum Yield Diversified Commodity Strategy ETF','PDBC',1),
    ('inflation asset','commodities','SPDR® Gold Shares','GLD',1),
    ('inflation asset','real estate','SPDR® Dow Jones® International Real Estate ETF','RWX',1),
],columns=['asset type','sub-asset type','Proxy Index','ETF','max_weight'])



constraint_state_street = {
    'conservative':{
        'sector_upper':{
            "equity":0.1,
            "fixed income":0.8,
            'inflation asset':0.1,
        },
        'sector_lower':{
            "equity":0.,
            "fixed income":0.,
            'inflation asset':0,
        }
    },
    'moderate conservative':{
        'sector_upper':{
            "equity":0.3,
            "fixed income":0.6,
            'inflation asset':0.1,
        },
        'sector_lower':{
            "equity":0.,
            "fixed income":0.,
            'inflation asset':0,
        }
    },
    'moderate':{
        'sector_upper':{
            "equity":0.5,
            "fixed income":0.4,
            'inflation asset':0.1,
        },
        'sector_lower':{
            "equity":0.,
            "fixed income":0.,
            'inflation asset':0.,
        }
    },
    'moderate growth':{
        'sector_upper':{
            "equity":0.65,
            "fixed income":0.25,
            'inflation asset':0.1,
        },
        'sector_lower':{
            "equity":0.,
            "fixed income":0.,
            'inflation asset':0.,
        }
    },
    'growth':{
        'sector_upper':{
            "equity":0.8,
            "fixed income":0.1,
            'inflation asset':0.1,
        },
        'sector_lower':{
            "equity":0.,
            "fixed income":0.,
            'inflation asset':0.,
        }
    },
    'maximum growth':{
        'sector_upper':{
            "equity":0.9,
            "fixed income":0,
            'inflation asset':0.1,
        },
        'sector_lower':{
            "equity":0.,
            "fixed income":0.,
            'inflation asset':0,
        }
    },
    
}