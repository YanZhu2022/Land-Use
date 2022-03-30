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


from land_use import lu_constants as const
from land_use.utils import file_ops as ops


class BaseYearLandUsePaths:
    """
    Path Class for the Base Year Land Use model.

    """
    # Constants

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
        self.write_folder = os.path.join(const.LU_FOLDER, const.BY_FOLDER, self.iter)
        self.log_folder = os.path.join(self.write_folder, const.LU_LOGGING_DIR)
        self.audit_folder = os.path.join(self.write_folder, const.LU_AUDIT_DIR)
        self.process_folder = os.path.join(self.write_folder, const.LU_PROCESS_DIR)
        self.output_folder = os.path.join(self.write_folder,  const.LU_OUTPUT_DIR)

        # Audits
        self.audit_fnames = {}
        for i in range(1, 12):
            self.audit_fnames[i] = const.AUDIT_FNAME % (i, self.base_year)

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
                                                            const.NOMIS_MYPE_MSOA_AGE_GENDER_PATH % self.base_year)
        self.geography_directory = os.path.join(self.import_folder, const.GEOGRAPHY_DIRECTORY)
        self.uk_2011_and_2021_la_path = os.path.join(self.geography_directory, const.UK_2011_AND_2021_LA_PATH)
        self.scottish_2011_z2la_path = os.path.join(self.geography_directory, const.SCOTTISH_2011_Z2LA_PATH)
        self.scottish_data_directory = os.path.join(self.import_folder, const.MYE_ONS_FOLDER % self.base_year,
                                                    const.MID_YEAR_MSOA % self.base_year)
        self.scottish_base_year_males_path = os.path.join(self.scottish_data_directory,
                                                          const.SCOTTISH_MALES_PATH % self.base_year)
        self.scottish_base_year_females_path = os.path.join(self.scottish_data_directory,
                                                            const.SCOTTISH_FEMALES_PATH % self.base_year)
        self.scottish_la_changes_post_2011_path = os.path.join(self.inputs_directory_mye,
                                                               const.SCOTTISH_LA_CHANGES_POST_2011_PATH)
        self.aps_ftpt_gender_base_year_path = os.path.join(self.import_folder, const.INPUTS_DIRECTORY_APS,
                                                           self.base_year, const.APS_FTPT_GENDER_PATH % self.base_year)
        self.nomis_base_year_mye_pop_by_la_path = os.path.join(self.inputs_directory_mye,
                                                               const.NOMIS_MYE_POP_BY_LA_PATH % self.base_year)
        self.aps_soc_path = os.path.join(self.import_folder, const.INPUTS_DIRECTORY_APS, self.base_year,
                                         const.APS_SOC_PATH % self.base_year)

        # Exports for step 3.2.5
        self.full_mye_aps_process_dir = os.path.join(self.process_folder, self.mye_pop_compiled_dir)
        self.hhr_pop_by_dag = os.path.join(self.audit_folder, self.mye_pop_compiled_dir, const.HHR_POP_BY_DAG %
                                           self.base_year)
        self.hhr_vs_all_pop_path = os.path.join(self.full_mye_aps_process_dir, const.HHR_VS_ALL_POP_FNAME
                                                % (self.model_zoning.lower(), self.base_year))
        self.full_la_level_adjustment_dir = os.path.join(self.process_folder, self.la_level_adjustment_dir)
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
        self.popoutput_path = os.path.join(self.process_folder, self.mye_pop_compiled_dir,
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
        self.verified_d_worker_path = os.path.join(self.output_folder, const.VERIFIED_D_WORKER_FNAME % self.base_year)
        self.verified_d_non_worker_path = os.path.join(self.output_folder,
                                                       const.VERIFIED_D_NON_WORKER_FNAME % self.base_year)

        # Exports for 3.2.10
        self.further_adjustments_dir_path = os.path.join(self.output_folder, self.further_adjustments_dir)
        self.hhpop_combined_path = os.path.join(self.output_folder, const.HHPOP_COMBINED_FNAME
                                                % self.base_year)
        self.final_zonal_hh_pop_by_t_fname = os.path.join(self.output_folder,
                                                          const.FINAL_ZONAL_HH_POP_BY_T_FNAME % self.base_year)
        self.all_pop_path = os.path.join(self.output_folder, const.ALL_POP_FNAME)
        self.all_pop_by_t_path = os.path.join(self.output_folder, const.ALL_POP_T_FNAME)
        # Imports for 3.2.11
        self.nomis_mye_base_year_path = os.path.join(self.inputs_directory_mye, const.NOMIS_MYE_PATH % self.base_year)

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
        self.audit_zonaltot_path = os.path.join(self.audit_folder, self.pop_with_full_dims_second_dir,
                                                const.AUDIT_ZONALTOT_FNAME % self.base_year)
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

