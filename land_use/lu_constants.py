"""
Constants from Land Use
"""

import os

import pandas as pd

PACKAGE_NAME = __name__.split('.')[0]

# SUFFIXES AND SEMI-STATIC CONFIG
COMPRESSION_SUFFIX = '.pbz2'
PROCESS_COUNT = -2

# PATHS
# Default land use folder
LU_FOLDER = 'I://NorMITs Land Use//'
BY_FOLDER = 'base_land_use'
FY_FOLDER = 'future_land_use'
DATA_FOLDER = 'Y://Data Strategy//Data//'

# Most recent Land Use Iteration
LU_MR_ITER = 'iter4h'
FYLU_MR_ITER = 'iter3c'
LU_IMPORTS = 'import'
LU_REFS = 'Lookups'

# Inputs
ZONE_NAME = 'MSOA'
ZONES_FOLDER = 'I:/NorMITs Synthesiser/Zone Translation/'
ZONE_TRANSLATION_PATH = ZONES_FOLDER + 'Export/msoa_to_lsoa/msoa_to_lsoa.csv'
ADDRESSBASE_PATH_LIST = 'I:/Data/AddressBase/2018/List of ABP datasets.csv'
# LU_FOLDER + '/' + LU_IMPORTS + '/AddressBase/2018/List of ABP datasets.csv'
KS401_PATH_FNAME = LU_FOLDER + '/' + LU_IMPORTS + '/' + 'Nomis Census 2011 Head & Household/KS401UK_LSOA.csv'
LU_AREA_TYPES = LU_FOLDER + '/area types/TfNAreaTypesLookup.csv'
ALL_RES_PROPERTY_PATH = 'I:/NorMITs Land Use/import/AddressBase/2018/processed'
CTripEnd_Database = 'I:/Data/NTEM/NTEM 7.2 outputs for TfN/'
LU_LOGGING_DIR = '00 Logging'
LU_PROCESS_DIR = '01 Process'
LU_AUDIT_DIR = '02 Audits'
LU_OUTPUT_DIR = '03 Outputs'

# Shapefile locations
DEFAULT_ZONE_REF_FOLDER = 'Y:/Data Strategy/GIS Shapefiles'
DEFAULT_LSOAREF = os.path.join(DEFAULT_ZONE_REF_FOLDER, 'UK LSOA and Data Zone Clipped 2011',
                               'uk_ew_lsoa_s_dz.csv')
DEFAULT_LADREF = os.path.join(DEFAULT_ZONE_REF_FOLDER, 'LAD GB 2017',
                              'Local_Authority_Districts_December_2017_Full_Clipped_Boundaries_in_Great_Britain.csv')
DEFAULT_MSOAREF = os.path.join(DEFAULT_ZONE_REF_FOLDER, 'UK MSOA and Intermediate Zone Clipped 2011',
                               'uk_ew_msoa_s_iz.csv')

# Path to a default land use build
RESI_LAND_USE_MSOA = os.path.join(
    LU_FOLDER,
    BY_FOLDER,
    LU_MR_ITER,
    'outputs',
    'land_use_output_safe_msoa.csv'
)

NON_RESI_LAND_USE_MSOA = os.path.join(
    LU_FOLDER,
    BY_FOLDER,
    LU_MR_ITER,
    'outputs',
    'land_use_2018_emp.csv')

E_CAT_DATA = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'HSL 2018',
    'non_freight_msoa_2018.csv'
)

UNM_DATA = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'SOC mix',
    'nomis_2021_10_12_165818.csv'
)

MSOA_REGION = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'msoa_region.csv'
)

# TODO: Doesn't exist yet
MSOA_SECTOR = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'msoa_sector.csv'
)

SCENARIO_FOLDERS = {'NTEM': 'SC00_NTEM',
                    'SC01_JAM': 'SC01_JAM',
                    'SC02_PP': 'SC02_PP',
                    'SC03_DD': 'SC03_DD',
                    'SC04_UZC': 'SC04_UZC'}

# TODO: Fill this in
SCENARIO_NUMBERS = {0: 'SC00_NTEM',
                    1: 'SC01_JAM'}

NTEM_POP_GROWTH = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'scenarios',
    'SC00_NTEM',
    'population',
    'future_population_growth.csv'
)

NTEM_EMP_GROWTH = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'scenarios',
    'SC00_NTEM',
    'employment',
    'future_workers_growth.csv'
)

# This is growth not values
NTEM_CA_GROWTH = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'scenarios',
    'SC00_NTEM',
    'car ownership',
    'ca_future_shares.csv'
)

NTEM_DEMOGRAPHICS_MSOA = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'scenarios',
    'SC00_NTEM',
    'demographics',
    'future_demographic_values.csv'
)

SOC_2DIGIT_SIC = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'SOC Mix',
    'soc_2_digit_sic_2018.csv'
)

SOC_BY_REGION = os.path.join(
    LU_FOLDER,
    LU_IMPORTS,
    'SOC Mix',
    'hsl_3cat_summary.csv'
)

# REFERENCES
# purposes to apply soc split to
SOC_P = [1, 2, 12]

# Property type dictionary: combines all flat types
PROPERTY_TYPE = {
    1: 1,  # detached
    2: 2,  # semi-detached
    3: 3,  # terrace
    4: 4,  # purpose-built flat
    5: 4,  # shared flat
    6: 4,  # flat in commercial
    7: 4,  # mobile home
    8: 8  # communal establishment
}

# Property type description to code
# TODO: lots of overlap with PROPERTY_TYPE, can it be a single object? (Used in main_build)
HOUSE_TYPE = {
    'Detached': 1,
    'Semi-detached': 2,
    'Terraced': 3,
    'Flat': 4
}

# NS-SeC category mapping
NS_SEC = {
    'NS-SeC 1-2': 1,
    'NS-SeC 3-5': 2,
    'NS-SeC 6-7': 3,
    'NS-SeC 8': 4,
    'NS-SeC L15': 5
}

# Car availabiity reference
CA_MODEL = pd.DataFrame({'cars': [0, 1, 2, 3],
                         'ca': [1, 2, 2, 2]})

REF_PATH = os.path.join(LU_FOLDER, LU_IMPORTS, LU_REFS)

AGE_REF = pd.read_csv(os.path.join(REF_PATH,
                                   'age_index.csv'))

GENDER_REF = pd.read_csv(os.path.join(REF_PATH,
                                      'gender_index.csv'))

HC_REF = pd.read_csv(os.path.join(REF_PATH,
                                  'household_composition_index.csv'))

# NTEM Traveller Type Reference
RAW_TT_INDEX = pd.read_csv(os.path.join(REF_PATH,
                                        'ntem_traveller_types.csv'))

TT_INDEX = pd.read_csv(os.path.join(REF_PATH,
                                    'ntem_traveller_types_normalised.csv'))

# TfN Traveller Type Reference
TFN_TT_INDEX = pd.read_csv(os.path.join(REF_PATH,
                                        'tfn_traveller_types_normalised.csv'),
                           dtype=int)

TFN_TT_DESC = pd.read_csv(os.path.join(REF_PATH,
                                       'tfn_traveller_types_illustrated.csv'))

# LU Pop Build Steps
BY_POP_BUILD_STEPS = [
    '3.2.1', '3.2.2', '3.2.3', '3.2.4', '3.2.5',
    '3.2.6', '3.2.7', '3.2.8', '3.2.9', '3.2.10',
    '3.2.11'
]

BY_POP_BUILD_STEP_DESCS = [
    'read in core property data',
    'filled property adjustment',
    'household occupancy adjustment',
    'property type mapping',
    '2018 MYPE uplift',
    'expand NTEM population and verify 1',
    'expand NTEM population and verify 2',
    'get subsets of worker and non-worker',
    'verify worker and non-worker',
    'adjust pop with full dimensions',
    'process CER data'
]

# Constants for Step 3.2.2
ZONE_TRANSLATION_COL = {'lsoa_zone_id': 'lsoaZoneID',
                        'msoa_zone_id': 'msoaZoneID'
                        }
UK_MSOA_COL = {'msoa11cd': 'msoaZoneID'}
FILLED_PROPS_COL = {'geography code': 'geography_code',
                    'Dwelling Type: All categories: Household spaces; measures: Value': 'Total_Dwells',
                    'Dwelling Type: Household spaces with at least one usual resident; measures: Value': 'Filled_Dwells'
                    }
CPT_DATA_COL = {'lsoa11cd': 'lsoaZoneID'}

# Exports for Step 3.2.2
FILLED_PROPERTIES_FNAME = 'gb_msoa_%s_dwells_occ.csv'
BALANCED_CPT_DATA_FNAME = 'gb_msoa_prt_%s_occupancy.csv'

CENSUS_COL_NEEDED = [2, 6, 7, 8, 10, 11, 12, 13]
CENSUS_COL_NAMES = ['geography_code', 'cpt1', 'cpt2', 'cpt3', 'cpt4', 'cpt5', 'cpt6', 'cpt7']

# Constants for Step 3.2.3
EWQS401 = 'QS401UK_LSOA.csv'
SQS401 = 'QS_401UK_DZ_2011.csv'
EWQS402 = 'QS402UK_LSOA.csv'
SQS402 = 'QS402UK_DZ_2011.csv'
ADDRESSBASE_EXTRACT_PATH = 'allResProperty%sClassified.csv'
ZONE_TRANSLATION_PATH_LAD_MSOA = os.path.join(ZONES_FOLDER, 'Export/lad_to_msoa/lad_to_msoa.csv')

# Constants for Step 3.2.5
MYE_POP_COMPILED_NAME = 'MYE_pop_compiled'
MODEL_YEAR = '2018'

MYE_MSOA_POP_NAME = 'gb_%s_%s_pop+hh_pop.csv'
MYE_ONS_FOLDER = 'MYE %s ONS'
POP_PROCESS_INPUTS = '%s_pop_process_inputs'
NOMIS_MYPE_MSOA_AGE_GENDER_PATH = 'nomis_%s_MYPE_MSOA_Age_Gender.csv'
GEOGRAPHY_DIRECTORY = 'Population Processing lookups'
UK_2011_AND_2021_LA_PATH = r'UK_2011_and_2021_LA_IDs.csv'
SCOTTISH_2011_Z2LA_PATH = r'2011_Scottish_Zones_to_LA.csv'
LOOKUP_GEOGRAPHY_2011_PATH = r'I:\NorMITs Land Use\import\2011 Census Micro lookups\geography.csv'
QS101_UK_PATH = r'I:\NorMITs Land Use\import\Nomis Census 2011 Head & Household\211022_QS101UK_ResidenstType_MSOA.csv'
MID_YEAR_MSOA = '%s_MidyearMSOA'
SCOTTISH_MALES_PATH = 'Males_Scotland_%s.csv'
SCOTTISH_FEMALES_PATH = 'Females_Scotland_%s.csv'
# Path with manual corrections to make proportions equal 1
LA_TO_MSOA_UK_PATH = r'I:\NorMITs Synthesiser\Zone Translation\Export\lad_to_msoa\lad_to_msoa_normalised.csv'
SCOTTISH_LA_CHANGES_POST_2011_PATH = r'ca11_ca19.csv'
APS_FTPT_GENDER_PATH = 'nomis_APS_FTPT_Gender_%s_only.csv'
INPUTS_DIRECTORY_APS = 'NOMIS APS'
NOMIS_MYE_POP_BY_LA_PATH = 'nomis_%s_MYE_LA_withareacodes_total_gender.csv'
APS_SOC_PATH = 'nomis_APS_SOC_%s.csv'
POP_SEG_FNAME = r'CTripEnd/Pop_Segmentations.csv'
CTripEnd = 'CTripEnd7_%d.accdb'
POP_DROP = ['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15',
            'K01', 'K02', 'K03', 'K04', 'K05', 'K06', 'K07', 'K08', 'K09', 'K10', 'K11', 'K12', 'K13', 'K14', 'K15']
POP_COL = ["I", "R", "B", "Borough", "ZoneID", "ZoneName"]
TZONEPOP_COL_RENAME = {"LZoneID": "ZoneID", "LBorough": "Borough", "LAreaType": "AreaType",
                       "LTravellerType": "TravellerType", "LIndivID": "IndivID"}
TZONEPOP_COL_DROP = ['GrowthinPeriod', 'GrowthperYear', 'GrowthtoYear', 'LPopulation', 'UPopulation', 'ZoneID_left',
                     'ZoneID_right', 'ZoneName_right', 'ZoneName_left', 'Borough_left', 'Borough_right', 'IndivID']
TZONEPOP_COLS_DROP = ['Population', 'ZoneID', 'overlap_population', 'ntem_population', 'msoa_population',
                      'overlap_msoa_pop_split_factor', 'overlap_type']
TZONEPOP_GROUPBY = ['msoaZoneID', 'AreaType', 'Borough', 'TravellerType', 'NTEM_TT_Name', 'Age_code', 'Age',
                    'Gender_code', 'Gender', 'Household_composition_code', 'Household_size', 'Household_car',
                    'Employment_type_code', 'Employment_type']
POPOUTPUT = 'ntem_gb_z_areatype_ntem_tt_%s_pop.csv'
NTEM_HH_POP_COLS = ['msoaZoneID', 'msoa11cd', 'Borough', 'TravellerType', 'NTEM_TT_Name', 'Age_code',
                    'Age', 'Gender_code', 'Gender', 'Household_composition_code', 'Household_size', 'Household_car',
                    'Employment_type_code', 'Employment_type', 'Population']

# Exports for 3.2.5
HHR_POP_BY_DAG = 'audit_3_gb_dag_%s_hh_pop.csv'
HHR_VS_ALL_POP_FNAME = 'gb_%s_%s_pop+hh_pop.csv'
HHR_WORKER_BY_D_FOR_EXPORT_FNAME = 'mye_gb_d_%s_wkrs_tot+by_ag.csv'
HHR_NONWORKER_BY_D_FOR_EXPORT_FNAME = 'mye_gb_d_%s_nwkrs_tot+by_ag.csv'
LA_INFO_FOR_2021_FNAME = r'lookup_gb_2021_lad_to_d.csv'
NTEM_LOGFILE_FNAME = 'NTEM_Pop_Interpolation_LogFile_%s.txt'
ZONE_PATH_FNAME = r'Export/ntem_to_msoa/ntem_msoa_pop_weighted_lookup.csv'
NTEM_HHPOP_TOTAL_FNAME = 'ntem_gb_z_%s_hh_pop.csv'
HHPOP_DT_TOTAL_FNAME = 'gb_msoa_%s_hh_pop.csv'
MYE_POP_COMPILED_FNAME = 'gb_msoa_agg_prt_%s_hh_pop.csv'
NTEM_HH_POP_FNAME = 'ntem_gb_msoa_ntem_tt_%s_mye_pop'

# Exports for 3.2.3
APPLY_HH_OCC_FNAME = 'resi_gb_%s_prt_%s_dwells+pop.csv'

# Imports for 3.2.3
HOPS_GROWTH_FNAME = 'HOPs/hops_growth_factors.csv'
# Constants for 3.2.4
CRP_COLS = ['ZoneID', 'census_property_type', 'UPRN', 'household_occupancy_18', 'population']
CRP_COL_RENAME = {'UPRN': 'Properties_from_3.2.3', 'population': 'Population_from_3.2.3'}
PROCESSED_CRP_COL_RENAME = {'UPRN': 'Properties_from_3.2.4', 'population': 'Population_from_3.2.4'}
CHECK_CRP_PARAMS = ['Properties', 'Population']
# Exports for 3.2.4
LU_FORMATTING_FNAME = 'resi_gb_%s_agg_prt_%s_dwells+pop.csv'

# Constants for 3.2.6 / 3.2.7
NORCOM_OUTPUT_MAIN_DIR = r'Base Year LU to NorCOM'
NorCOM_NTEM_HHpop_col = ['msoa11cd', 'lu_TravellerType', 'NorCOM_result']
NTEM_HHPOP_COLS = ['msoaZoneID', 'msoa11cd', 'TravellerType', 'Age_code', 'Gender_code', 'Household_composition_code',
                   'Household_size', 'Household_car', 'Employment_type_code']
NTEM_HHPOP_COL_RENAME = {'msoaZoneID': 'z',
                         'TravellerType': 'ntem_tt',
                         'Age_code': 'a',
                         'Gender_code': 'g',
                         'Household_composition_code': 'h',
                         'Employment_type_code': 'e',
                         'NorCOM_result': 'P_NTEM'}
CENSUS_F_VALUE_FNAME = '2011 Census Furness/04 Post processing/Outputs/NorMITs_2011_post_ipfn_f_values.csv'
NORMITS_HHPOP_BYDT_COL_RENAME = {'ZoneID': 'MSOA',
                                 'population': 'crp_P_t',
                                 'UPRN': 'properties',
                                 'census_property_type': 't'}
NTEM_HHPOP_BYDT_COL_RENAME = {'P_aghetns': 'P_t'}
HHPOP_COL_RENAME = {'P_aghetns': 'NTEM_HH_pop', 'P_aghetns_aj': 'people'}

ZONE_2021LA_COLS = ['NorMITs Zone', '2021 LA', '2021 LA Name']
HHPOP_OUTPUT_COLS = ['2021_LA_code', '2021_LA_Name', 'z', 'MSOA', 'a', 'g', 'h', 'e', 't', 'n', 's', 'properties',
                     'people']
SEG_FOLDER = 'NTEM Segmentation Audits'

# Imports for 3.2.6 / 3.2.7
INPUT_NTEM_HHPOP_FNAME = r'NorCOM outputs\%s\NorCOM_TT_output.csv'
ZONE_2021LA_FNAME = 'Lookups/MSOA_1991LA_2011LA_2021LA_LAgroups.csv'

# Exports for 3.2.6 / 3.2.7
OUTPUT_NTEM_HHPOP_FNAME = 'output_0_ntem_gb_msoa_ntem_tt_%s_aj_hh_pop.csv'
POP_TRIM_WITH_FULL_DIMS_FNAME = 'gb_%s_tfn_tt_agg_prt_%s_pop'
NTEM_HHPOP_BYDT_FNAME = 'ntem_gb_z_t_%s_hh_pop.csv'
POP_WITH_FULL_DIMS_FNAME = 'gb_lad_%s_tfn_tt_agg_prt_%s_properties+hh_pop'
GB_AVE_HH_OCC_FNAME = 'gb_t_%s_ave_hh_occ.csv'

# Constants for 3.2.8
HHPOP_GROUPBY = ['2021_LA_code', 'a', 'g', 'h', 'e', 't', 'n', 's']
SEED_WORKER_COLS = ['2021_LA_code', 'ge', 's', 'a', 'h', 't', 'n', 'people']
GE_COMBINATION_VALUES = ['1', '2', '3', '4']

# Exports for 3.2.8
HHPOP_NWKRS_AG_LA_FNAME = 'audit_8_dag_%s_nwkrs.csv'
HHPOP_WKRS_GE_LA_FNAME = 'audit_9_dge_%s_wkrs.csv'
HHPOP_WKRS_S_LA_FNAME = 'audit_10_ds_%s_wkrs.csv.csv'

# Constants for 3.2.9
GE = {
    'Male fte': 1,
    'Male pte': 2,
    'Female fte': 3,
    'Female pte': 4
}

S = {
    'higher': 1,
    'medium': 2,
    'skilled': 3
}

A = {
    'Children': 1,
    'M_16-74': 2,
    'F_16-74': 2,
    'M_75 and over': 3,
    'F_75 and over': 3
}

G = {
    'Children': 1,
    'M_16-74': 2,
    'F_16-74': 3,
    'M_75 and over': 2,
    'F_75 and over': 3
}

SEED_WORKER_FNAME = 'seed_gb_d_tfn_tt_agg_prt_%s_wkrs.csv'
AJ_HHPOP_WORKERS_LA_COLS = ['2021_LA_code', 'a', 'g', 'h', 'e', 't', 'n', 's', 'total']
WRKS_AJ_FACTOR_LA_COLS = ['2021_LA_code', 'a', 'g', 'h', 'e', 't', 'n', 's', 'wkr_aj_factor']
NWKRS_AJ_FACTOR_LA = ['2021_LA_code', 'a', 'g', 'nwkr_aj_factor']
SEG_TO_TT_COLS = ['a', 'g', 'h', 'e', 'n', 's']

# Exports for 3.2.9
FURNESSED_DATA_FNAME = 'furnessed_gb_d_tfn_tt_agg_prt_%s_wkrs.csv'
NWKRS_AJ_FACTOR_LA_FNAME = 'gb_lad_ag_%s_nwkrs_aj_factor.csv'
WKRS_AJ_FACTOR_LA_FNAME = 'gb_lad_tfn_tt_agg_prt_%s_wkrs_aj_factor.csv'
VERIFIED_D_WORKER_FNAME = 'output_1_resi_gb_lad_tfn_tt_agg_prt_%s_wkrs'
VERIFIED_D_NON_WORKER_FNAME = 'output_2_resi_gb_lad_tfn_tt_agg_prt_%s_nwkrs'

# Imports for 3.2.9
NORMITS_SEG_TO_TFN_TT_FNAME = r'Lookups\NorMITs_segments_to_TfN_tt\normits_segs_to_tfn_tt.csv'

# Imports for 3.2.10
NOMIS_MYE_PATH = 'nomis_MYE_%s.csv'

# Constants for 3.2.10
CER_POP_EXPANDED_COLS = ['MSOA', 'Zone', 'a', 'g', 'h', 'e', 't', 'n', 's', '2021_LA_Name', 'zaghetns_CER']

# Exports for 3.2.10
HHPOP_COMBINED_FNAME = 'output_3_resi_gb_lad_msoa_tfn_tt_agg_prt_%s_hh_pop'
FINAL_ZONAL_HH_POP_BY_T_FNAME = 'output_4_resi_gb_msoa_agg_prt_%s_hh_pop+dwells+hh_occ.csv'
ALL_POP_FNAME = 'output_6_resi_gb_msoa_tfn_tt_prt_%s_pop'
ALL_POP_T_FNAME = 'output_7_resi_gb_msoa_tfn_tt_%s_pop'

# Exports for 3.2.11
CER_POP_EXPANDED_FNAME = 'output_5_gb_msoa_tfn_tt_%s_CER_pop'

AUDIT_FNAME = 'Audit_3.2.%d_%s.txt'
HPA_FOLDER = 'audit_1_hops_population_audits_%s'
HPA_FNAME = 'audit_1-1_hops_gb_%s_%s_pop.csv'
AUDIT_LU_FORMATTING_FNAME = 'audit_2_gb_%s_agg_prt_%s_dwells+pop.csv'
AUDIT_3_2_5_CSV = 'audit_4_mye_ntem_crp_comparison_%s_pop.csv'
AUDIT_3_2_6_CSV = 'audit_5_gb_msoa_check_mye_%s_pop_vs_tfn_tt_agg_prt_%s_pop.csv'
AUDIT_6_FOLDER = 'audit_6_ntem_segmentation_audits'
AUDIT_ZONALTOT_FNAME = 'audit_7_gb_msoa_check_mype_ntem_normits_%s_hh_pop.csv'
SEGMENTS = {'t': [1, 'agg_prt'],
            'Household_car': [2, 'car_avail'],
            'Household_size': [3, 'hh_size'],
            'h': [4, 'hh_comp'],
            'a': [5, 'age'],
            'g': [6, 'gender'],
            'e': [7, 'employment'],
            }
SEG_FNAME = "audit_6-%d_ntem_gb_msoa_%s_%s_pop_check.csv"
AUDIT_3_2_8_COLS = WRKS_AJ_FACTOR_LA_COLS[:-1]
AUDIT_3_2_8_DATA_EXPORT_FNAME = 'audit_11_gb_msoa_tfn_tt_agg_prt_%s_hh_pop+wkrs+nwkrs.csv'
AUDIT_HHPOP_BY_D_FNAME = 'audit_12_gb_d_%s_pop_deviation.csv'
AUDIT_HHPOP_BY_DAG_FNAME = 'audit_13_gb_dag_%s_pop_deviation.csv'
HHPOP_COMBINED_CHECK_LA_FNAME = 'audit_15_gb_dag_%s_check_furnessed_hh_pop.csv'
HHPOP_COMBINED_CHECK_Z_FNAME = 'audit_16_gb_msoa_%s_check_furnessed_hh_pop.csv'
HHPOP_COMBINED_PDIFF_EXTREMES_FNAME = 'audit_14_gb_msoa_%s_min_max_hh_pop_percent_diff.csv'
CHECK_ALL_POP_BY_D_FNAME = 'audit_17_gb_lad_%s_check_pop.csv'
