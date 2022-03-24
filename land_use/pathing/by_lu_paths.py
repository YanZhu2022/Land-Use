# -*- coding: utf-8 -*-
"""
Created on: Mon February 21, 2022
Updated on:

Original author: Nirmal Kumar
Last update made by:
Other updates made by:

File purpose:
Classes which build all the paths for base year land use inputs and outputs
"""
# Builtins
import os

# Local imports
import numpy as np
import pandas as pd

import land_use as lu
from land_use import lu_constants as const
from land_use.utils import timing
from land_use.utils import file_ops as ops
from land_use.utils import general as gen
from land_use.utils import compress


class BaseYearLandUsePaths:
    """
    Path Class for the Base Year Land Use model.

    """
    # Constants
    _outputs = 'outputs'
    _reports = 'reports'
    _audits = 'Audits'
    _pop_write = 'land_use_%s_pop.csv'
    _emp_write = 'land_use_%s_emp.csv'
    _nomis_census_path = 'Nomis Census 2011 Head & Household'

    # Define the names of the export dirs
    copy_address_database_dir = '3.2.1_read_in_core_property_data'
    filled_properties_dir = '3.2.2_filled_property_adjustment'
    apply_household_occupancy_dir = '3.2.3_apply_household_occupancy'
    land_use_formatting_dir = '3.2.4_land_use_formatting'
    mye_pop_compiled_dir = '3.2.5_uplifting_base_year_pop_base_year_MYPE'
    pop_with_full_dims_dir = '3.2.6_expand_NTEM_pop'
    pop_with_full_dims_second_dir = '3.2.7_verify_population_profile_by_dwelling_type'
    subsets_worker_nonworker_dir = '3.2.8_subsets_of_workers+nonworkers'
    la_level_adjustment_dir = '3.2.9_verify_district_level_worker_and_nonworker'
    further_adjustments_dir = '3.2.10_adjust_zonal_pop_with_full_dimensions'
    cer_dir = '3.2.11_process_CER_data'

    def __init__(self,
                 iteration: str,
                 base_year: str,
                 census_year: str,
                 model_zoning: str,
                 ):
        """
        Builds the export paths for base year land use model
        Parameters
        ----------
        """
        # Init

        self.iter = iteration
        self.base_year = base_year
        self.census_year = census_year
        self.model_zoning = model_zoning
        self.import_folder = os.path.join(const.LU_FOLDER, const.LU_IMPORTS)  # Path is I:\NorMITs Land Use\import
        self.write_folder = os.path.join(const.LU_FOLDER, const.BY_FOLDER, self.iter,
                                         self._outputs)  # Path is I:\NorMITs Land Use\base_land_use\iter\outputs
        self.audit_folder = os.path.join(const.LU_FOLDER, const.BY_FOLDER, self.iter, const.LU_AUDIT_DIR)
        self.process_folder = os.path.join(const.LU_FOLDER, const.BY_FOLDER, self.iter, const.LU_PROCESS_DIR)
        self.output_folder = os.path.join(const.LU_FOLDER, const.BY_FOLDER, self.iter, const.LU_OUTPUT_DIR)
        self.report_folder = os.path.join(self.write_folder, self._reports)
        self.pop_write_name = os.path.join(self.write_folder, self._pop_write % str(self.base_year))
        self.emp_write_name = os.path.join(self.write_folder, self._emp_write % str(self.base_year))

        # Audits
        self.audit_fnames = {}
        for i in range(1, 12):
            self.audit_fnames[i] = const.AUDIT_FNAME % (i, self.base_year)

        # # Set object paths
        # self.out_paths = {
        #     'write_folder': self.write_folder,
        #     'report_folder': self.report_folder,
        #     'pop_write_path': self.pop_write_name,
        #     'emp_write_path': self.emp_write_name
        # }

        self.filled_properties_path = os.path.join(self.process_folder, self.filled_properties_dir,
                                                   const.FILLED_PROPERTIES_FNAME % self.base_year)

        self.bal_cpt_data_path = os.path.join(self.process_folder, self.apply_household_occupancy_dir,
                                              const.BALANCED_CPT_DATA_FNAME % self.census_year)
        # Imports census household data
        self.census_dat = os.path.join(self.import_folder, self._nomis_census_path)

        # Imports for step 3.2.2
        self.ks401_path = os.path.join(self.import_folder, const.KS401_PATH_FNAME)

        # Exports for step 3.2.3
        self.apply_hh_occ_path = os.path.join(self.process_folder, self.apply_household_occupancy_dir,
                                              const.APPLY_HH_OCC_FNAME % (self.model_zoning.lower(), self.base_year))
        self.crp_cols = ['ZoneID', 'census_property_type', 'UPRN', 'household_occupancy_%s' % self.base_year[-2:],
                         'population']
        # Imports for step 3.2.3
        self.hops_path = os.path.join(self.import_folder, const.HOPS_GROWTH_FNAME)
        self.addressbase_extract_path = os.path.join(const.ALL_RES_PROPERTY_PATH, const.ADDRESSBASE_EXTRACT_PATH
                                                     % self.model_zoning)
        # Exports for step 3.2.4
        self.lu_formatting_path = os.path.join(self.process_folder, self.land_use_formatting_dir,
                                               const.LU_FORMATTING_FNAME % (self.model_zoning.lower(), self.base_year))
        # Imports for step 3.2.5
        self.mye_msoa_pop = os.path.join(self.process_folder, self.mye_pop_compiled_dir,
                                         const.MYE_MSOA_POP_NAME % (self.model_zoning.lower(), self.base_year))
        self.inputs_directory_mye = os.path.join(self.import_folder,
                                                 const.MYE_ONS_FOLDER % self.base_year,
                                                 const.POP_PROCESS_INPUTS % self.base_year)
        self.nomis_mype_msoa_age_gender_path = os.path.join(self.inputs_directory_mye,
                                                            const.NOMIS_MYPE_MSOA_AGE_GENDER_PATH)
        self.geography_directory = os.path.join(self.import_folder, const.GEOGRAPHY_DIRECTORY)
        self.uk_2011_and_2021_la_path = os.path.join(self.geography_directory, const.UK_2011_AND_2021_LA_PATH)
        self.scottish_2011_z2la_path = os.path.join(self.geography_directory, const.SCOTTISH_2011_Z2LA_PATH)
        self.scottish_data_directory = os.path.join(self.import_folder, const.MYE_ONS_FOLDER % self.base_year,
                                                    const.MID_YEAR_MSOA % self.base_year)
        self.scottish_base_year_males_path = os.path.join(self.scottish_data_directory,
                                                          const.SCOTTISH_MALES_PATH)
        self.scottish_base_year_females_path = os.path.join(self.scottish_data_directory,
                                                            const.SCOTTISH_FEMALES_PATH)
        self.scottish_la_changes_post_2011_path = os.path.join(self.inputs_directory_mye,
                                                               const.SCOTTISH_LA_CHANGES_POST_2011_PATH)
        self.aps_ftpt_gender_base_year_path = os.path.join(self.import_folder, const.INPUTS_DIRECTORY_APS,
                                                           const.APS_FTPT_GENDER_PATH)
        self.nomis_base_year_mye_pop_by_la_path = os.path.join(self.inputs_directory_mye,
                                                               const.NOMIS_MYE_POP_BY_LA_PATH)
        self.aps_soc_path = os.path.join(self.import_folder, const.INPUTS_DIRECTORY_APS,
                                         const.APS_SOC_PATH % self.base_year)

        # Exports for step 3.2.5
        self.full_mye_aps_process_dir = os.path.join(self.process_folder, self.mye_pop_compiled_dir)
        self.hhr_pop_by_dag = os.path.join(self.audit_folder, self.mye_pop_compiled_dir, const.HHR_POP_BY_DAG %
                                           self.base_year)
        self.hhr_vs_all_pop_path = os.path.join(self.full_mye_aps_process_dir, const.HHR_VS_ALL_POP_FNAME
                                                % (self.model_zoning.lower(), self.base_year))
        self.full_la_level_adjustment_dir = os.path.join(self.write_folder, self.la_level_adjustment_dir)
        self.hhr_worker_by_d_for_export_path = os.path.join(self.full_la_level_adjustment_dir,
                                                            const.HHR_WORKER_BY_D_FOR_EXPORT_FNAME % self.base_year)
        self.hhr_nonworker_by_d_for_export_path = os.path.join(self.full_la_level_adjustment_dir,
                                                               const.HHR_NONWORKER_BY_D_FOR_EXPORT_FNAME %
                                                               self.base_year)
        self.la_info_for_2021_path = os.path.join(self.full_la_level_adjustment_dir,
                                                  const.LA_INFO_FOR_2021_FNAME)
        self.ntem_logfile_path = os.path.join(self.audit_folder, self.mye_pop_compiled_dir,
                                              const.NTEM_LOGFILE_FNAME % self.base_year)
        self.zone_path = os.path.join(const.ZONES_FOLDER, const.ZONE_PATH_FNAME)
        self.pop_seg_path = os.path.join(self.import_folder, const.POP_SEG_FNAME)
        self.popoutput_path = os.path.join(self.write_folder, self.mye_pop_compiled_dir,
                                           const.POPOUTPUT % self.base_year)
        self.ntem_hhpop_total_path = os.path.join(self.process_folder, self.mye_pop_compiled_dir,
                                                  const.NTEM_HHPOP_TOTAL_FNAME % self.base_year)

        self.hhpop_dt_total_path = os.path.join(self.process_folder, self.mye_pop_compiled_dir,
                                                const.HHPOP_DT_TOTAL_FNAME % self.base_year)
        self.mye_pop_compiled_path = os.path.join(self.process_folder, self.mye_pop_compiled_dir,
                                                  const.MYE_POP_COMPILED_FNAME % self.base_year)
        self.ntem_hh_pop_path = os.path.join(self.process_folder, self.mye_pop_compiled_dir,
                                             const.NTEM_HH_POP_FNAME % self.base_year)

        # Imports for 3.2.6 / 3.2.7
        self.input_ntem_hhpop_path = os.path.join(self.import_folder, const.INPUT_NTEM_HHPOP_FNAME % self.base_year)
        self.census_f_value_path = os.path.join(self.import_folder, const.CENSUS_F_VALUE_FNAME)

        # Exports for 3.2.6 / 3.2.7
        self.norcom_output_main_dir = os.path.join(self.import_folder, const.NORCOM_OUTPUT_MAIN_DIR, self.iter)
        if not os.path.exists(self.norcom_output_main_dir):
            ops.create_folder(self.norcom_output_main_dir)
        self.output_ntem_hhpop_filepath = os.path.join(self.norcom_output_main_dir, const.OUTPUT_NTEM_HHPOP_FNAME
                                                       % self.base_year)
        self.output_ntem_hhpop_out_path = os.path.join(self.output_folder, const.OUTPUT_NTEM_HHPOP_FNAME
                                                       % self.base_year)

        self.pop_trim_with_full_dims_path = os.path.join(self.process_folder, self.pop_with_full_dims_dir,
                                                         const.POP_TRIM_WITH_FULL_DIMS_FNAME
                                                         % (self.model_zoning.lower(), self.base_year))
        self.ntem_hhpop_bydt_path = os.path.join(self.process_folder, self.pop_with_full_dims_dir,
                                                 const.NTEM_HHPOP_BYDT_FNAME % self.base_year)
        self.zone_2021la_path = os.path.join(self.import_folder, const.ZONE_2021LA_FNAME)

        # Exports for 3.2.6 / 3.2.7
        self.seg_audit = os.path.join(self.audit_folder, self.pop_with_full_dims_second_dir, const.SEG_FOLDER)
        self.pop_with_full_dims_path = os.path.join(self.process_folder, self.pop_with_full_dims_second_dir,
                                                    const.POP_WITH_FULL_DIMS_FNAME
                                                    % (self.model_zoning.lower(), self.base_year))
        self.gb_ave_hh_occ_path = os.path.join(self.process_folder, self.pop_with_full_dims_dir,
                                               const.GB_AVE_HH_OCC_FNAME % self.base_year)

        # Exports for 3.2.8
        self.subsets_worker_nonworker_dir_path = os.path.join(self.audit_folder,
                                                              self.subsets_worker_nonworker_dir)
        self.hhpop_nwkrs_ag_la_path = os.path.join(self.subsets_worker_nonworker_dir_path,
                                                   const.HHPOP_NWKRS_AG_LA_FNAME)
        self.hhpop_wkrs_ge_la_path = os.path.join(self.subsets_worker_nonworker_dir_path,
                                                  const.HHPOP_WKRS_GE_LA_FNAME)
        self.hhpop_wkrs_s_la_path = os.path.join(self.subsets_worker_nonworker_dir_path,
                                                 const.HHPOP_WKRS_S_LA_FNAME)

        # Imports for 3.2.9
        self.normits_seg_to_tfn_tt = os.path.join(self.import_folder, const.NORMITS_SEG_TO_TFN_TT_FNAME)
        # Exports for 3.2.9
        self.seed_worker_path = os.path.join(self.audit_folder, self.la_level_adjustment_dir, const.SEED_WORKER_FNAME
                                             % self.base_year)
        self.seed_path = os.path.join(self.process_folder, self.la_level_adjustment_dir, const.FURNESSED_DATA_FNAME
                                      % self.base_year)
        self.nwkrs_aj_factor_la_path = os.path.join(self.process_folder, self.la_level_adjustment_dir,
                                                    const.NWKRS_AJ_FACTOR_LA_FNAME % self.base_year)
        self.wkrs_aj_factor_la_path = os.path.join(self.process_folder, self.la_level_adjustment_dir,
                                                   const.WKRS_AJ_FACTOR_LA_FNAME % self.base_year)
        self.verified_d_worker_path = os.path.join(self.output_folder, self.la_level_adjustment_dir,
                                                   const.VERIFIED_D_WORKER_FNAME % self.base_year)
        self.verified_d_non_worker_path = os.path.join(self.output_folder, self.la_level_adjustment_dir,
                                                       const.VERIFIED_D_NON_WORKER_FNAME % self.base_year)

        # Exports for 3.2.10
        self.further_adjustments_dir_path = os.path.join(self.output_folder, self.further_adjustments_dir)
        self.hhpop_combined_path = os.path.join(self.further_adjustments_dir_path, const.HHPOP_COMBINED_FNAME
                                                % self.base_year)
        self.final_zonal_hh_pop_by_t_fname = os.path.join(self.further_adjustments_dir_path,
                                                          const.FINAL_ZONAL_HH_POP_BY_T_FNAME % self.base_year)
        self.all_pop_path = os.path.join(self.further_adjustments_dir_path, const.ALL_POP_FNAME)
        self.all_pop_by_t_path = os.path.join(self.further_adjustments_dir_path, const.ALL_POP_T_FNAME)
        # Imports for 3.2.11
        self.nomis_mye_base_year_path = os.path.join(self.inputs_directory_mye, const.NOMIS_MYE_PATH)

        # Exports for 3.2.11
        self.cer_pop_expanded_path = os.path.join(self.output_folder, const.CER_POP_EXPANDED_FNAME
                                                  % self.base_year)

        # Audit for 3.2.1
        self.audit_3_2_1_path = os.path.join(self.audit_folder, self.copy_address_database_dir, self.audit_fnames[1])
        self.audit_3_2_2_path = os.path.join(self.audit_folder, self.filled_properties_dir, self.audit_fnames[2])
        self.hpa_folder = os.path.join(self.audit_folder, self.apply_household_occupancy_dir,
                                       const.HPA_FOLDER % self.base_year)
        if not os.path.exists(self.hpa_folder):
            ops.create_folder(self.hpa_folder)
        self.arp_msoa_audit_path = os.path.join(self.hpa_folder, const.HPA_FNAME % (self.model_zoning.lower(),
                                                                                    self.base_year))
        self.audit_3_2_3_path = os.path.join(self.hpa_folder, self.audit_fnames[3])
        self.audit_land_use_formatting_path = os.path.join(self.audit_folder, self.land_use_formatting_dir,
                                                           const.AUDIT_LU_FORMATTING_FNAME % (
                                                               self.model_zoning.lower(), self.base_year))
        self.audit_3_2_4_path = os.path.join(self.audit_folder, self.land_use_formatting_dir, self.audit_fnames[4])
        self.audit_3_2_5_csv_path = os.path.join(self.audit_folder, self.mye_pop_compiled_dir,
                                                 const.AUDIT_3_2_5_CSV % self.base_year)
        self.audit_3_2_5_path = os.path.join(self.audit_folder, self.mye_pop_compiled_dir, self.audit_fnames[5])
        self.audit_3_2_6_csv_path = os.path.join(self.audit_folder, self.pop_with_full_dims_dir,
                                                 const.AUDIT_3_2_6_CSV % (self.base_year, self.base_year))
        self.audit_3_2_6_path = os.path.join(self.audit_folder, self.pop_with_full_dims_dir, self.audit_fnames[6])
        self.audit_6_folder = os.path.join(self.audit_folder, self.pop_with_full_dims_second_dir,
                                           const.AUDIT_6_FOLDER)
        if not os.path.exists(self.audit_6_folder):
            ops.create_folder(self.audit_6_folder)
        self.audit_zonaltot_path = os.path.join(self.audit_6_folder, const.AUDIT_ZONALTOT_FNAME % self.base_year)
        self.audit_3_2_7_path = os.path.join(self.audit_folder, self.pop_with_full_dims_second_dir,
                                             self.audit_fnames[7])
        self.audit_3_2_8_data_export_path = os.path.join(self.audit_folder, self.subsets_worker_nonworker_dir,
                                                         const.AUDIT_3_2_8_DATA_EXPORT_FNAME % self.base_year)
        self.audit_3_2_8_path = os.path.join(self.audit_folder, self.subsets_worker_nonworker_dir, self.audit_fnames[8])
        self.audit_hhpop_by_d_path = os.path.join(self.audit_folder, self.la_level_adjustment_dir,
                                                  const.AUDIT_HHPOP_BY_D_FNAME % self.base_year)
        self.audit_hhpop_by_dag_path = os.path.join(self.audit_folder, self.la_level_adjustment_dir,
                                                    const.AUDIT_HHPOP_BY_DAG_FNAME % self.base_year)
        self.audit_3_2_9_path = os.path.join(self.audit_folder, self.la_level_adjustment_dir, self.audit_fnames[9])
        self.hhpop_combined_check_z_path = os.path.join(self.audit_folder, self.further_adjustments_dir,
                                                        const.HHPOP_COMBINED_CHECK_Z_FNAME % self.base_year)
        self.hhpop_combined_check_la_path = os.path.join(self.audit_folder, self.further_adjustments_dir,
                                                         const.HHPOP_COMBINED_CHECK_LA_FNAME % self.base_year)
        self.hhpop_combined_pdiff_extremes_path = os.path.join(self.audit_folder, self.further_adjustments_dir,
                                                               const.HHPOP_COMBINED_PDIFF_EXTREMES_FNAME %
                                                               self.base_year)
        self.audit_further_adjustments_directory = os.path.join(self.audit_folder, self.further_adjustments_dir)
        self.audit_3_2_10_path = os.path.join(self.audit_folder, self.further_adjustments_dir, self.audit_fnames[10])
        self.check_all_pop_by_d_path = os.path.join(self.audit_folder, self.further_adjustments_dir,
                                                    const.CHECK_ALL_POP_BY_D_FNAME % self.base_year)
        self.audit_cer_directory = os.path.join(self.audit_folder, self.cer_dir)
        self.audit_3_2_11_10_path = os.path.join(self.audit_folder, self.further_adjustments_dir, self.audit_fnames[11])
        self.audit_3_2_11_path = os.path.join(self.audit_folder, self.cer_dir, self.audit_fnames[11])

    def audit_3_2_1(self) -> None:
        # TODO(NK): Write doc string
        out_lines = [
            '### Audit for Step 3.2.1 ###',
            'Created: %s', str(timing.get_datetime()),
            'Step 3.2.1 currently does nothing, so there is nothing to audit',
        ]
        with open(self.audit_3_2_1_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_2(self) -> None:
        # TODO(NK): Write doc string
        out_lines = [
            '### Audit for Step 3.2.2 ###',
            'Created: %s', str(timing.get_datetime()),
            'Step 3.2.2 currently has no audits listed, so there is nothing to audit',
        ]
        with open(self.audit_3_2_2_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_3(self, all_res_property: pd.DataFrame) -> None:
        # TODO(NK): Write doc string

        arp_msoa_audit = all_res_property.groupby('ZoneID')['population'].sum().reset_index()
        gen.safe_dataframe_to_csv(arp_msoa_audit, self.arp_msoa_audit_path, index=False)
        arp_msoa_audit_total = arp_msoa_audit['population'].sum()
        out_lines = [
            '### Audit for Step 3.2.3 ###',
            'Created: %s', str(timing.get_datetime()),
            'The total arp population is currently: %s', str(arp_msoa_audit_total),
            'A zonal breakdown of the arp population has been created here:',
            self.arp_msoa_audit_path,
        ]
        with open(self.audit_3_2_3_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_4(self,
                    crp: pd.DataFrame,
                    crp_for_audit: pd.DataFrame,
                    processed_crp_for_audit: pd.DataFrame,
                    txt1: str = "",
                    ) -> None:
        # TODO(NK): Write doc string

        crp_for_audit = crp_for_audit.groupby(['ZoneID'])[['UPRN', 'population']].sum()
        crp_for_audit = crp_for_audit.rename(columns=const.CRP_COL_RENAME)
        processed_crp_for_audit = processed_crp_for_audit.groupby(['ZoneID'])[['UPRN', 'population']].sum()
        processed_crp_for_audit = processed_crp_for_audit.rename(columns=const.PROCESSED_CRP_COL_RENAME)
        crp_for_audit = pd.merge(crp_for_audit, processed_crp_for_audit, how='left', on='ZoneID')
        crp_for_audit['Check_Properties'] = (crp_for_audit['Properties_from_3.2.4'] - crp_for_audit[
            'Properties_from_3.2.3']) / crp_for_audit['Properties_from_3.2.3']
        crp_for_audit['Check_Population'] = (crp_for_audit['Population_from_3.2.4'] - crp_for_audit[
            'Population_from_3.2.3']) / crp_for_audit['Population_from_3.2.3']

        gen.safe_dataframe_to_csv(crp_for_audit, self.audit_land_use_formatting_path, index=False)

        for p in const.CHECK_CRP_PARAMS:
            full_param = 'Check_%s' % p
            txt = ['For %s:' % p, 'Min percentage variation is: {}%'.format(str(crp_for_audit[full_param].min() * 100)),
                   'Max percentage variation is: {}%'.format(str(crp_for_audit[full_param].max() * 100)),
                   'Mean percentage variation is: {}%'.format(str(crp_for_audit[full_param].mean() * 100))]
            txt1 = txt1 + '\n'.join(txt)

        out_lines = [
            '### Audit for Step 3.2.4 ###',
            'Created: %s', str(timing.get_datetime()),
            'The total number of properties at the end of this step is: %s' % str(crp.UPRN.sum()),
            'The total population at the end of this step is: %s' % str(crp.population.sum()),
            'The Step 3.2.4 zonal breakdown of properties and population'
            ' has been checked against Step 3.2.3.',
            txt1,
            'A full listing of the zonal breakdown of population and properties has been created here:',
            self.audit_land_use_formatting_path,
            'These should match those found in the output of Step 3.2.3.',
        ]
        with open(self.audit_3_2_4_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_5(self,
                    audit_aj_crp: pd.DataFrame,
                    audit_ntem_hhpop: pd.DataFrame,
                    audit_mye_msoa_pop: pd.DataFrame,
                    mye_msoa_pop: pd.DataFrame,
                    ) -> None:

        # TODO(NK): Write doc string
        audit_aj_crp = audit_aj_crp[['ZoneID', 'population']]
        audit_aj_crp = audit_aj_crp.groupby(['ZoneID'])['population'].sum().reset_index()
        audit_aj_crp = audit_aj_crp.rename(columns={'population': 'crp_pop', 'ZoneID': 'MSOA'})
        audit_ntem_hhpop = audit_ntem_hhpop[['msoa11cd', 'pop_aj']]
        audit_ntem_hhpop = audit_ntem_hhpop.groupby(['msoa11cd'])['pop_aj'].sum().reset_index()
        audit_ntem_hhpop = audit_ntem_hhpop.rename(columns={'pop_aj': 'NTEM_pop', 'msoa11cd': 'MSOA'})
        audit_mye_msoa_pop = audit_mye_msoa_pop[['MSOA', 'Total_HHR']]
        audit_mye_msoa_pop = audit_mye_msoa_pop.rename(columns={'Total_HHR': 'MYE_pop'})
        audit_3_2_5_csv = pd.merge(audit_mye_msoa_pop, audit_ntem_hhpop, how='left', on='MSOA')
        audit_3_2_5_csv = pd.merge(audit_3_2_5_csv, audit_aj_crp, how='left', on='MSOA')
        audit_3_2_5_csv['MYE_vs_NTEM'] = (audit_3_2_5_csv['MYE_pop'] -
                                          audit_3_2_5_csv['NTEM_pop']) / audit_3_2_5_csv['NTEM_pop']
        audit_3_2_5_csv['NTEM_vs_crp'] = (audit_3_2_5_csv['NTEM_pop'] -
                                          audit_3_2_5_csv['crp_pop']) / audit_3_2_5_csv['crp_pop']
        audit_3_2_5_csv['crp_vs_MYE'] = (audit_3_2_5_csv['crp_pop'] -
                                         audit_3_2_5_csv['MYE_pop']) / audit_3_2_5_csv['MYE_pop']
        audit_3_2_5_csv_max = max(audit_3_2_5_csv['MYE_vs_NTEM'].max(),
                                  audit_3_2_5_csv['NTEM_vs_crp'].max(),
                                  audit_3_2_5_csv['crp_vs_MYE'].max())
        audit_3_2_5_csv_min = min(audit_3_2_5_csv['MYE_vs_NTEM'].min(),
                                  audit_3_2_5_csv['NTEM_vs_crp'].min(),
                                  audit_3_2_5_csv['crp_vs_MYE'].min())
        audit_3_2_5_csv_mean = np.mean([audit_3_2_5_csv['MYE_vs_NTEM'].mean(),
                                        audit_3_2_5_csv['NTEM_vs_crp'].mean(),
                                        audit_3_2_5_csv['crp_vs_MYE'].mean()])
        audit_3_2_5_csv.to_csv(self.audit_3_2_5_csv_path, index=False)

        out_lines = [
            '### Audit for Step 3.2.5 ###',
            'Created: %s', str(timing.get_datetime()),
            'The total %s population from MYPE is: %s' % (self.base_year, str(mye_msoa_pop.Total_Pop.sum())),
            'The total %s household population from MYPE is: %s' % (self.base_year, str(mye_msoa_pop.Total_HHR.sum())),
            'The total %s household population output from Step 3.2.5 is: ' % self.base_year,
            '   By zone, age, gender, HH composition and employment status(from NTEM): %s' %
            str(audit_ntem_hhpop['NTEM_pop'].sum()),
            '   By zone and dwelling type: %s' % str(audit_aj_crp['crp_pop'].sum()),
            'Comparing zonal HH population. All %age difference values should be very close to 0',
            'The max, min and mean of the three possible comparisons are presented here:',
            '   Max percentage difference: {}%'.format(str(audit_3_2_5_csv_max * 100)),
            '   Min percentage difference: {}%'.format(str(audit_3_2_5_csv_min * 100)),
            '   Mean percentage difference: {}%'.format(str(audit_3_2_5_csv_mean * 100)),
            'A full zonal breakdown of these metrics is presented in:',
            self.audit_3_2_5_csv_path,
        ]
        with open(self.audit_3_2_5_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_6(self,
                    audit_ntem_hhpop_trim: pd.DataFrame,
                    audit_original_hhpop: pd.DataFrame,
                    ntem_hhpop_trim: pd.DataFrame,
                    ) -> None:

        # TODO(NK): Write doc string

        audit_ntem_hhpop_trim = audit_ntem_hhpop_trim[['msoa11cd', 'P_aghetns']]
        audit_ntem_hhpop_trim = audit_ntem_hhpop_trim.groupby(['msoa11cd'])['P_aghetns'].sum().reset_index()
        audit_3_2_6_csv = pd.merge(audit_original_hhpop, audit_ntem_hhpop_trim,
                                   how='left', left_on='MSOA', right_on='msoa11cd')
        audit_3_2_6_csv['HH_pop_%age_diff'] = (audit_3_2_6_csv['P_aghetns'] -
                                               audit_3_2_6_csv['MYE_pop']) / audit_3_2_6_csv['MYE_pop']
        audit_3_2_6_csv_max = audit_3_2_6_csv['HH_pop_%age_diff'].max()
        audit_3_2_6_csv_min = audit_3_2_6_csv['HH_pop_%age_diff'].min()
        audit_3_2_6_csv_mean = audit_3_2_6_csv['HH_pop_%age_diff'].mean()
        gen.safe_dataframe_to_csv(audit_3_2_6_csv, self.audit_3_2_6_csv_path, index=False)

        out_lines = [
            '### Audit for Step 3.2.6 ###',
            'Created: %s', str(timing.get_datetime()),
            'The total %s population from MYPE is: %s' % (self.base_year, str(ntem_hhpop_trim.P_aghetns.sum())),
            'Comparing zonal HH population original to present:',
            '   Max percentage difference: {}%'.format(str(audit_3_2_6_csv_max * 100)),
            '   Min percentage difference: {}%'.format(str(audit_3_2_6_csv_min * 100)),
            '   Mean percentage difference: {}%'.format(str(audit_3_2_6_csv_mean * 100)),
            'All of the above should be equal (or very close) to 0.',
            'A full zonal breakdown of these metrics is presented in:',
            self.audit_3_2_6_csv_path,
        ]
        with open(self.audit_3_2_6_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_7(self,
                    hhpop: pd.DataFrame,
                    audit_original_hhpop: pd.DataFrame,
                    aj_crp: pd.DataFrame,
                    ) -> None:

        # TODO(NK): Write doc string
        zonaltot = hhpop.groupby(['z', 'MSOA'])[['people', 'NTEM_HH_pop']].sum().reset_index()
        zonaltot = zonaltot.rename(columns={'people': 'NorMITs_Zonal', 'NTEM_HH_pop': 'NTEM_Zonal'})
        audit_zonaltot = pd.merge(audit_original_hhpop, zonaltot, how='right', on='MSOA')
        audit_zonaltot['HH_pop_%age_diff'] = (audit_zonaltot['NorMITs_Zonal'] -
                                              audit_zonaltot['MYE_pop']) / audit_zonaltot['MYE_pop']
        audit_zonaltot_max = audit_zonaltot['HH_pop_%age_diff'].max()
        audit_zonaltot_min = audit_zonaltot['HH_pop_%age_diff'].min()
        audit_zonaltot_mean = audit_zonaltot['HH_pop_%age_diff'].mean()
        gen.safe_dataframe_to_csv(audit_zonaltot, self.audit_zonaltot_path, index=False)

        def seg_audit(seg: str,
                      numbr: int,
                      seg_fname: str,
                      ) -> None:
            df = hhpop.groupby(['z', 'MSOA', seg])[['people', 'NTEM_HH_pop']].sum().reset_index()
            df_check = df.merge(zonaltot, how='left', on=['z'])
            df_check['Ab_Perdiff'] = df_check['people'] / df_check['NTEM_HH_pop'] - 1
            df_check['NorMITs_profile'] = df_check['people'] / df_check['NorMITs_Zonal']
            df_check['NTEM_profile'] = df_check['NTEM_HH_pop'] / df_check['NTEM_Zonal']
            df_check['Profile_Perdiff'] = df_check['NorMITs_profile'] / df_check['NTEM_profile'] - 1
            gen.safe_dataframe_to_csv(df_check, os.path.join(self.audit_6_folder,
                                                             (const.SEG_FNAME % (numbr, seg_fname, self.base_year))),
                                      index=False)

        for segment, fname in const.SEGMENTS.items():
            seg_audit(seg=segment, numbr=fname[0], seg_fname=fname[1])

        out_lines = [
            '### Audit for Step 3.2.7 ###',
            'Created: %s', str(timing.get_datetime()),
            '>>> IMPORTANT NOTE <<<',
            '   If you can\'t find the output you are looking for, which should have been',
            '   exported to Step 3.2.7, try looking in the Step 3.2.6 directory, as both',
            '   steps run using the same function, so separating the outputs can be tricky!',
            'High level summaries:',
            '   The total HH pop from aj_ntem is: %s' % str(hhpop.people.sum()),
            '   The total HH pop from aj_crp is: %s' % str(aj_crp.population.sum()),
            'The zonal variation in HH pop against the original MYPE derived HH pop has:',
            '   Max percentage difference: {}%'.format(str(audit_zonaltot_max * 100)),
            '   Min percentage difference: {}%'.format(str(audit_zonaltot_min * 100)),
            '   Mean percentage difference: {}%'.format(str(audit_zonaltot_mean * 100)),
            'These percentage differences should be equal (or close to) 0.',
            'A full zonal breakdown of these differences can be found here:',
            self.audit_zonaltot_path,
            'Additionally, number of segmentation audits have also been produced.',
            'These can be found in:',
            self.audit_6_folder,
            'Again, the differences are expected to be small.'
        ]

        with open(self.audit_3_2_7_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_8(self,
                    audit_3_2_8_data: pd.DataFrame,
                    audit_hhpop_workers_la: pd.DataFrame,
                    audit_hhpop_non_workers_la: pd.DataFrame,
                    hhpop_workers: pd.DataFrame,
                    hhpop_non_workers: pd.DataFrame,
                    ) -> None:

        # TODO(NK): Do doc strings
        audit_3_2_8_data = pd.merge(audit_3_2_8_data, audit_hhpop_workers_la,
                                    how='left', on=const.AUDIT_3_2_8_COLS)
        audit_3_2_8_data['people'] = audit_3_2_8_data['people'].fillna(0)
        audit_3_2_8_data = audit_3_2_8_data.rename(columns={'people': 'worker_pop'})
        audit_3_2_8_data = pd.merge(audit_3_2_8_data, audit_hhpop_non_workers_la,
                                    how='left', on=const.AUDIT_3_2_8_COLS)
        audit_3_2_8_data['people'] = audit_3_2_8_data['people'].fillna(0)
        audit_3_2_8_data = audit_3_2_8_data.rename(columns={'people': 'non_worker_pop'})
        audit_3_2_8_data['worker+non_worker_pop'] = audit_3_2_8_data['worker_pop'] + audit_3_2_8_data['non_worker_pop']
        audit_3_2_8_data['Check_pop_tots'] = (audit_3_2_8_data['worker+non_worker_pop'] -
                                              audit_3_2_8_data['HHpop']) / audit_3_2_8_data['HHpop']
        audit_3_2_8_data_max = audit_3_2_8_data['Check_pop_tots'].max()
        audit_3_2_8_data_min = audit_3_2_8_data['Check_pop_tots'].min()
        audit_3_2_8_data_mean = audit_3_2_8_data['Check_pop_tots'].mean()
        compress.write_out(audit_3_2_8_data, self.audit_3_2_8_data_export_path)
        out_lines = [
            '### Audit for Step 3.2.8 ###',
            'Created: %s', str(timing.get_datetime()),
            'Totals at the end of Step 3.2.8:',
            '   Workers: {}'.format(hhpop_workers.people.sum()),
            '   Non_workers: {}'.format(hhpop_non_workers.people.sum()),
            '   Population (worker + non-worker): {}'.format(
                hhpop_workers.people.sum() + hhpop_non_workers.people.sum()),
            'Also check variations at the LA level (by a, g, h, e, t, n, s), where:',
            '   Max %age difference is:' + str(audit_3_2_8_data_max * 100) + '%',
            '   Min %age difference is:' + str(audit_3_2_8_data_min * 100) + '%',
            '   Mean %age difference is:' + str(audit_3_2_8_data_mean * 100) + '%',
            'These differences should be 0 by definition.',
            'A compressed full d, a, g, h, e, t, n, s breakdown is included for completeness.',
            'It is expected that a csv dump would have been too big.',
            'The compressed file is dumped here (plus its file extension):',
            self.audit_3_2_8_data_export_path,
        ]
        with open(self.audit_3_2_8_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_9(self,
                    audit_hhpop_by_d: pd.DataFrame,
                    aj_hhpop_non_workers_la: pd.DataFrame,
                    pe_df: pd.DataFrame,
                    pe_dag_for_audit: pd.DataFrame,
                    hhpop_workers_la: pd.DataFrame,
                    hhpop_nwkrs_ag_la: pd.DataFrame,
                    ) -> None:

        # TODO(NK): Do doc strings
        audit_hhpop_by_d = audit_hhpop_by_d.append(aj_hhpop_non_workers_la)
        audit_hhpop_by_dag = audit_hhpop_by_d.copy()
        audit_hhpop_by_d = audit_hhpop_by_d.groupby('2021_LA_code')['total'].sum().reset_index()
        audit_hhpop_by_d = audit_hhpop_by_d.rename(columns={'total': 'Step_3.2.9_total_pop'})
        audit_hhpop_by_d = pd.merge(audit_hhpop_by_d, pe_df, how='left', on='2021_LA_code')
        audit_hhpop_by_d['%age_diff_in_pop'] = (audit_hhpop_by_d['Step_3.2.9_total_pop'] -
                                                audit_hhpop_by_d['Total_HHR']) / audit_hhpop_by_d['Total_HHR']
        audit_hhpop_by_d_max = audit_hhpop_by_d['%age_diff_in_pop'].max()
        audit_hhpop_by_d_min = audit_hhpop_by_d['%age_diff_in_pop'].min()
        audit_hhpop_by_d_mean = audit_hhpop_by_d['%age_diff_in_pop'].mean()
        gen.safe_dataframe_to_csv(audit_hhpop_by_d, self.audit_hhpop_by_d_path, index=False)

        audit_hhpop_by_dag = audit_hhpop_by_dag.groupby(['2021_LA_code', 'a', 'g'])['total'].sum().reset_index()
        audit_hhpop_by_dag = audit_hhpop_by_dag.rename(columns={'total': 'Step_3.2.9_total_pop'})
        audit_hhpop_by_dag = pd.merge(audit_hhpop_by_dag, pe_dag_for_audit, how='left', on=['2021_LA_code', 'a', 'g'])
        audit_hhpop_by_dag['%age_diff_in_pop'] = (audit_hhpop_by_dag['Step_3.2.9_total_pop'] -
                                                  audit_hhpop_by_dag['HHR_pop']) / audit_hhpop_by_dag['HHR_pop']
        audit_hhpop_by_dag_max = audit_hhpop_by_dag['%age_diff_in_pop'].max()
        audit_hhpop_by_dag_min = audit_hhpop_by_dag['%age_diff_in_pop'].min()
        audit_hhpop_by_dag_mean = audit_hhpop_by_dag['%age_diff_in_pop'].mean()
        gen.safe_dataframe_to_csv(audit_hhpop_by_dag, self.audit_hhpop_by_dag_path, index=False)

        out_lines = [
            '### Audit for Step 3.2.9 ###',
            'Created: %s', str(timing.get_datetime()),
            'The total %s population is currently: %s' % (self.base_year, str(hhpop_workers_la.total.sum() +
                                                                              hhpop_nwkrs_ag_la.nonworker.sum())),
            'Total HHR workers currently {}'.format(hhpop_workers_la.total.sum()),
            'Total non_workers currently {}'.format(hhpop_nwkrs_ag_la.nonworker.sum()),
            'Comparing LA level HH population original to present (worker + non-worker):',
            '   By LA only:',
            '       Max percentage difference: {}%'.format(str(audit_hhpop_by_d_max * 100)),
            '       Min percentage difference: {}%'.format(str(audit_hhpop_by_d_min * 100)),
            '       Mean percentage difference: {}%'.format(str(audit_hhpop_by_d_mean * 100)),
            '   By LA, age and gender:',
            '       Max percentage difference: {}%'.format(str(audit_hhpop_by_dag_max * 100)),
            '       Min percentage difference: {}%'.format(str(audit_hhpop_by_dag_min * 100)),
            '       Mean percentage difference: {}%'.format(str(audit_hhpop_by_dag_mean * 100)),
            'All of the above should be equal (or very close) to 0.',
            'A full breakdown of the LA only data is presented in:',
            self.audit_hhpop_by_d_path,
            'A full breakdown of the LA, age and gender data is presented in:',
            self.audit_hhpop_by_dag_path
        ]
        with open(self.audit_3_2_9_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_10(self,
                     hhpop_combined_check_z: pd.DataFrame,
                     hhpop_combined_check_la: pd.DataFrame,
                     hhpop_combined: pd.DataFrame,
                     hhpop_combined_pdiff_extremes: pd.DataFrame,
                    ) -> None:

        # TODO(NK): Do doc strings

        gen.safe_dataframe_to_csv(hhpop_combined_check_z, self.hhpop_combined_check_z_path, index=False)
        gen.safe_dataframe_to_csv(hhpop_combined_check_la, self.hhpop_combined_check_la_path, index=False)
        out_lines = [
            '### Audit for Step 3.2.10 ###',
            'Created: %s', str(timing.get_datetime()),
            'The total %s population is currently: %s'% (self.base_year,str(hhpop_combined.people.sum())),
            'Comparing z and d level HH population original to present:',
            '   By zone (z) - values will vary from 0. See Tech Note for expected values:',
            '       Max percentage difference: {}%'.format(str(hhpop_combined_check_z['percentage_diff'].max() * 100)),
            '       Min percentage difference: {}%'.format(str(hhpop_combined_check_z['percentage_diff'].min() * 100)),
            '       Mean percentage difference: {}%'.format(str(hhpop_combined_check_z['percentage_diff'].mean() * 100)),
            '   By LA (d) - values should not vary significantly from 0:',
            '       Max percentage difference: {}%'.format(str(hhpop_combined_check_la['percentage_diff'].max() * 100)),
            '       Min percentage difference: {}%'.format(str(hhpop_combined_check_la['percentage_diff'].min() * 100)),
            '       Mean percentage difference: {}%'.format(str(hhpop_combined_check_la['percentage_diff'].mean() * 100)),
            'A full zonal breakdown of the data is presented in:',
            self.hhpop_combined_check_z_path,
            'A full district breakdown of the data is presented in:',
            self.hhpop_combined_check_la_path,
        ]
        gen.safe_dataframe_to_csv(hhpop_combined_pdiff_extremes, self.hhpop_combined_pdiff_extremes_path)

        with open(self.audit_3_2_10_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_11_10(self,
                        all_pop_by_d: pd.DataFrame,
                        check_all_pop_by_d: pd.DataFrame,
                        ) -> None:

        # TODO(NK): Do doc strings

        out_lines = [
            '### Audit for the parts of Step 3.2.11 carried out directly by Step 3.2.10 ###',
            'Created: %s', str(timing.get_datetime()),
            'Note that the audit of CER pop is carried out in the Step 3.2.11 Audit directory.',
            'The total %s population at the end of the running process is:',
               str(all_pop_by_d['people'].sum()),
            'Checking final district total population against MYE district population:',
            '   The min percentage difference: {}%'.format(str(check_all_pop_by_d['pop_deviation'].min() * 100)),
            '   The max percentage difference: {} % '.format(str(check_all_pop_by_d['pop_deviation'].max() * 100)),
            '   The mean percentage difference: {}%'.format(str(check_all_pop_by_d['pop_deviation'].mean() * 100)),
            'The overall deviation is %s people' % str(check_all_pop_by_d['people'].sum() - check_all_pop_by_d[
                                                       'MYE_pop'].sum()),
            'All of the above values should be equal (or close) to 0.',
            'A full breakdown of the %s population by d can be found at:' % self.base_year,
            self.check_all_pop_by_d_path,
            'The Step 3.2.11 Audits directory is located here:',
            self.audit_cer_directory,
            'The Step 3.2.11 main audit file should be obvious in it.',
        ]

        with open(self.audit_3_2_11_10_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))

    def audit_3_2_11(self,
                     cer_pop_expanded: pd.DataFrame,
                     ) -> None:

        # TODO(NK): Do doc strings
        out_lines = [
            '### Audit for Step 3.2.10 ###',
            'Created: %s', str(timing.get_datetime()),
            'The total CER population is: %s' % str(cer_pop_expanded.zaghetns_CER.sum()),
            'All other logging for this step is carried out in the Step 3.2.10 audit directory.',
            'This is because the script for Step 3.2.10 carries out all of the processing that',
            'is assigned to Step 3.2.11 in the Technical Note beyond the creation of CER pop.',
            'The Step 3.2.10 Audits directory is located here:',
            self.audit_further_adjustments_directory,
            'The Step 3.2.11 (in Step 3.2.10) main audit file should be obvious in it.'
        ]
        with open(self.audit_3_2_11_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))