"""
Created on: -
Updated on: Friday March 25th 2022

Original author: -
Last update made by: Nirmal Kumar


File purpose:
Base year Land Use Model for NorMITS
"""
# Allow class self type hinting
from __future__ import annotations

# Builtins
import os
import warnings
import pyodbc
from ipfn import ipfn

# Third party imports
import pandas as pd
import numpy as np

# local imports
import land_use as lu
from land_use import lu_constants as const

from land_use.utils import file_ops
from land_use.utils import timing
from land_use.utils import compress
from land_use.utils import general as gen
from land_use.audits import audits

from land_use.base_land_use import employment
from land_use.pathing.by_lu_paths import BaseYearLandUsePaths


class BaseYearLandUse(BaseYearLandUsePaths):
    _log_fname = "%s_base_year_land_use.log"
    _running_report_fname = 'running_parameters.txt'

    # Constants
    __version__ = lu.__version__

    def __init__(self,
                 norcom_status: str,
                 base_year: str,
                 iteration: str = const.LU_MR_ITER,
                 model_zoning: str = const.ZONE_NAME,
                 zones_folder: str = const.ZONES_FOLDER,
                 zone_translation_path=const.ZONE_TRANSLATION_PATH,
                 ks401path=const.KS401_PATH_FNAME,
                 area_type_path=const.LU_AREA_TYPES,
                 ctripend_database_path=const.CTripEnd_Database,
                 emp_e_cat_data_path=const.E_CAT_DATA,
                 emp_soc_cat_data_path=const.SOC_BY_REGION,
                 emp_unm_data_path=const.UNM_DATA,
                 scenario_name: str = None,
                 census_year: str = '2011',
                 export_process_files: bool = False,
                 ):
        """
        Sets up and validates arguments for the Base year Land Use model

        Parameters
        ----------
        norcom_status: str
            Whether to import from NorCOM or export to NorCOM.

        base_year: str
            The year for which the land use model has to run.

        iteration: str = const.LU_MR_ITER
            The name of this iteration of the Base year Land Use model.

        model_zoning: str = const.ZONE_NAME
            Zoning system to be considered in the model.

        scenario_name: str
            Name of the scenario.

        census_year: str = '2011'
            Census year to be considered for the land use model.

        export_process_files: bool = True
            Whether to export all the process files or not.
            Not to be confused with outputs. They get exported by default.
        """

        # File ops
        # self.model_folder = model_folder + '/' + output_folder
        # self.iteration = iteration
        # self.home_folder = model_folder + '/' + output_folder + '/' + iteration
        # self.import_folder = model_folder + '/' + import_folder + '/'

        # Resi inputs
        self.zones_folder = zones_folder
        self.zone_translation_path = zone_translation_path
        self.ks401path = ks401path
        self.area_type_path = area_type_path
        self.CTripEnd_Database_path = ctripend_database_path

        # Non resi inputs
        self.e_cat_emp_path = emp_e_cat_data_path
        self.soc_emp_path = emp_soc_cat_data_path
        self.unemp_path = emp_unm_data_path

        # Basic config
        self.model_zoning = model_zoning
        self.base_year = base_year
        self.scenario_name = scenario_name.upper() if scenario_name is not None else ''
        self.norcom = norcom_status
        self.export_process_files = export_process_files

        # Set up superclass
        super().__init__(
            iteration=iteration,
            base_year=base_year,
            census_year=census_year,
            model_zoning=model_zoning
        )
        # Build paths
        print('Building Base Year paths and preparing to run')

        # TODO: Implement this way of checking state
        self.step_keys = const.BY_POP_BUILD_STEPS

        self.step_descs = const.BY_POP_BUILD_STEP_DESCS

        # self._check_state()
        # '03 Outputs' will also be a directory at this level,
        # but does not have the step sub-directories

        # Build folders
        if not os.path.exists(self.write_folder):
            file_ops.create_folder(self.write_folder)
        for folder_type in const.LIST_OF_TYPE_FOLDERS:
            for listed_folder in const.LIST_OF_STEP_FOLDERS:
                if not os.path.exists(os.path.join(self.write_folder, folder_type, listed_folder)):
                    file_ops.create_folder(os.path.join(self.write_folder, folder_type, listed_folder))
        if not os.path.exists(self.output_folder):
            file_ops.create_folder(self.output_folder)
        if not os.path.exists(self.log_folder):
            file_ops.create_folder(self.log_folder)

        # Establish a state dictionary recording which steps have been run
        # These are aligned with the section numbers in the documentation
        # TODO: enable a way to init from a point part way through the process

        # Create a logger
        logger_name = "%s.%s" % (lu.get_package_logger_name(), self.__class__.__name__)
        log_file_path = os.path.join(self.log_folder, self._log_fname % self.base_year)
        self._logger = lu.get_logger(
            logger_name=logger_name,
            log_file_path=log_file_path,
            instantiate_msg="Initialised New Base Year Land Use Model %s" % self.base_year,
        )
        self._write_running_report()

    def _write_running_report(self):
        """
        Outputs a simple report detailing inputs and outputs
        """
        # Define the lines to output
        out_lines = [
            'Code Version: %s' % str(lu.__version__),
            'Base Year Land Use Iteration: %s' % str(self.iter),
            'Base Year: %s' % str(self.base_year),
            'Census Year: %s' % str(self.census_year),
            '',
            '### Other information ###',
            'NorCOM status: %s' % self.norcom,
            'Export_process_files: %s' % self.export_process_files,

            'import_folder: %s' % self.import_folder,
            'log_folder: %s' % self.log_folder,
            'audit_folder: %s' % self.audit_folder,
            'process_folder: %s' % self.process_folder,
            'output_folder: %s' % self.output_folder,
        ]

        # Write out to disk
        output_path = os.path.join(self.write_folder, self._running_report_fname)
        with open(output_path, 'w') as out:
            out.write('\n'.join(out_lines))

    def build_by_pop(self,
                     verbose: bool = True,
                     ) -> None:
        # TODO: Method name, this is more of an adjustment to a base now
        """
        Runs the base year land use model by population.

        Parameters
        ----------
        verbose: bool = True
            Whether to print the text or not.

        Returns
        -------
        None

        """
        # Initialise timing
        start_time = timing.current_milli_time()
        self._logger.info("Starting %s base year land use model" % self.base_year)
        # TODO: Check that this picks up the 2011 process!

        # TODO: Change from copy to check imports
        # Run through the Base Year Build process
        # Steps from main build
        # TODO: Decide if this is used for anything anymore

        self._logger.info("\n" + "=" * 150)
        self._logger.info("Running step 3.2.1, reading in core property data")
        print('\n' + '=' * 75)
        self.copy_addressbase_files(verbose=verbose)

        self._logger.info('\n' + '=' * 150)
        self._logger.info('Running step 3.2.2, calculating the filled property adjustment factors')
        print('\n' + '=' * 75)
        filled_prop_df = self.filled_properties(verbose=verbose)

        self._logger.info('\n' + '=' * 150)
        print('\n' + '=' * 75)
        self._logger.info('Running step 3.2.3, household occupancy adjustment')
        all_res_property = self.apply_household_occupancy(prob_filled=filled_prop_df, verbose=verbose)

        self._logger.info('\n' + '=' * 150)
        print('\n' + '=' * 75)
        self._logger.info('Running step 3.2.4, combining flat types')
        self.property_type_refactor(crp=all_res_property, verbose=verbose)

        self._logger.info('\n' + '=' * 150)
        print('\n' + '=' * 75)
        self._logger.info('Running step 3.2.5, uplifting 2018 population according to 2018 MYPE')
        compiled_mye_pop = self.mye_pop_compiled(crp=all_res_property)

        self._logger.info('\n' + '=' * 150)
        print('\n' + '=' * 75)
        self._logger.info('Running step 3.2.6, expand NTEM population to full dimensions')
        self._logger.info('Also running step 3.2.7, verify population profile by dwelling type')
        self.pop_with_full_dimensions(compiled_mye_pop, verbose=verbose)

        self._logger.info('\n' + '=' * 150)
        print('\n' + '=' * 75)
        self._logger.info('Running step 3.2.8, get subsets of worker and non-worker')
        self._logger.info('Called from "census_lu.py" so saving outputs to files')
        self._logger.info('but not saving any variables to memory')
        self._logger.info('Note that this function will get called again by other functions')
        self.subsets_worker_nonworker('census_and_by_lu')

        self._logger.info('\n' + '=' * 150)
        print('\n' + '=' * 75)
        self._logger.info('Running step 3.2.9, verify district level worker and non-worker')
        self.la_level_adjustment()

        self._logger.info('\n' + '=' * 150)
        print('\n' + '=' * 75)
        self._logger.info('Running step 3.2.10, adjust zonal pop with full dimensions')
        self.adjust_zonal_workers_nonworkers(compiled_mye_pop[0])

        end_time = timing.current_milli_time()
        time_taken = timing.time_taken(start_time, end_time)
        self._logger.info("%s Base Year Land Use Model took:%s" % (self.base_year, time_taken))

        # Step 3.2.11 should always be called from Step 3.2.10 (to save read/writing massive files)
        # Syntax for calling it is maintained here (albeit commented out) for QA purposes
        # if self.state['3.2.11 process CER data'] == 0:
        #     logging.info('')
        #     logging.info('=========================================================================')
        #     print('\n' + '=' * 75)
        #     logging.info('Running step 3.2.11, process CER data')
        #     BaseYear2018_population_process.process_cer_data(self, hhpop_combined_from_3_2_10, la_2_z_from_3_2_10)

    def copy_addressbase_files(self,
                               verbose=True
                               ) -> None:
        # TODO: Should this be deprecated now?
        """
        Previously the module was copying the relevant ABP files
        from import drive to self.write_folder for use in later functions.
        Currently does nothing.

        Parameters
        ----------
        verbose: bool = True
            Whether to print the text or not.

        Returns
        -------
        None
        """
        self._logger.info('Running Step 3.2.1')
        gen.print_w_toggle('Running Step 3.2.1', verbose=verbose)

        # dest = self.home_folder
        # files = gen.safe_read_csv(const.ADDRESSBASE_PATH_LIST)
        gen.print_w_toggle('no longer copying into default iter folder', verbose=verbose)

        # for file in files.FilePath:
        #    try:
        #        shutil.copy(file, dest)
        #        print("Copied over file into default iter folder: " + file)
        #    except IOError:
        #        print("File not found")

        # Audit
        audits.audit_3_2_1(self)

        self._logger.info('Step 3.2.1 completed')
        gen.print_w_toggle('Step 3.2.1 completed', verbose=verbose)

    def read_zt_file(self) -> pd.DataFrame:
        """
        Reads the zone translation file and renames columns.

        Parameters:
        -----------
        None

        Returns:
        --------
        zone_translation:
            Returns the read zone translation file with renamed columns.
        """
        zone_trans = gen.safe_read_csv(const.ZONE_TRANSLATION_PATH, usecols=const.ZONE_TRANSLATION_COL.keys())
        zone_trans = zone_trans.rename(columns=const.ZONE_TRANSLATION_COL)
        return zone_trans

    def filled_properties(self,
                          verbose: bool = True,
                          ) -> pd.DataFrame:

        """
        Aims to adjust for the unoccupied properties.

        Parameters:
        -----------
        verbose: bool = True
            Whether to print text or not.

        Returns:
        --------
        filled_properties_df:
            Dataframe containing the probability of dwellings occupied in each zone.

        """
        self._logger.info('Running Step 3.2.2')
        gen.print_w_toggle('Running Step 3.2.2', verbose=verbose)
        # Define folder name for outputs

        # Reads in the census filled property data
        filled_properties_df = gen.safe_read_csv(self.ks401_path, usecols=const.FILLED_PROPS_COL.keys())
        filled_properties_df = filled_properties_df.rename(columns=const.FILLED_PROPS_COL)

        # Reads in the zone translation (default LSOA to MSOA)
        zone_translation = self.read_zt_file()

        # Merges and applies the zone translation onto the census data
        filled_properties_df = filled_properties_df.rename(columns={'geography_code': 'lsoaZoneID'})
        filled_properties_df = filled_properties_df.merge(zone_translation, on='lsoaZoneID')
        filled_properties_df = filled_properties_df.drop(columns='lsoaZoneID')
        filled_properties_df = filled_properties_df.groupby('msoaZoneID').sum().reset_index()

        # Calculate the probability that a property is filled
        cols = list(filled_properties_df)
        del cols[0]
        filled_properties_df['Prob_DwellsFilled'] = (filled_properties_df[cols[1]] /
                                                     filled_properties_df[cols[0]])
        filled_properties_df = filled_properties_df.drop(columns=cols)

        # The above filled properties probability is based on E+W so need to join back to Scottish MSOAs

        uk_msoa = gen.safe_read_csv(const.DEFAULT_MSOAREF, usecols=['msoa11cd']).rename(columns=const.UK_MSOA_COL)
        filled_properties_df = uk_msoa.merge(filled_properties_df, on='msoaZoneID', how="outer")
        filled_properties_df = filled_properties_df.fillna(1)  # default to all Scottish properties being occupied
        # Adam - DONE, we need to think how to organise the structure of outputs files per step

        gen.safe_dataframe_to_csv(filled_properties_df, self.filled_properties_path, index=False)
        # Audit
        audits.audit_3_2_2(self)

        self._logger.info('Step 3.2.2 completed')
        gen.print_w_toggle('Step 3.2.2 completed', verbose=verbose)
        return filled_properties_df

    # Sub-function used by apply_household_occupancy. Not called directly by census_lu.py
    def lsoa_census_data_prep(self,
                              dat_path: lu.PathLike,
                              population_tables: list[str],
                              property_tables: list[str],
                              geography: lu.PathLike,
                              verbose: bool = True,
                              ) -> pd.DataFrame:
        """
        This function prepares the census data by picking fields out of the census csvs.
        Computes the ratio between the population and number of properties
        to return the household occupancy.

        Parameters
        ----------
        dat_path: lu.PathLike
            Location of the census data.

        population_tables: list[str]
            list of the csv file names for population data.

        property_tables: list[str]
            list of the csv file names for property data.

        geography: lu.PathLike
            File path for geometry, defaults to LSOA.

        verbose: bool = True
            Whether to print text or not.

        Returns
        -------
        household_occupancy:
            Population data, property data and geometry merged into a single DataFrame.
        """
        self._logger.info('Running lsoa_census_data_prep function')
        gen.print_w_toggle('Running lsoa_census_data_prep function', verbose=verbose)

        def _read_census_table(census_tables: list[str],
                               table_type: str,
                               ) -> pd.DataFrame:
            """
            Read census table and convert it to dataframe

            Parameters
            ----------
            census_tables: list[str]
                A list of strings contains the census data filenames.

            table_type: str
                Contains the column heading for the data.
                The string value is 'population' or 'properties'.

            Returns
            -------
            imports:
                Returns the census data as dataframe
            """

            self._logger.info('Running _read_census_table sub-function')
            gen.print_w_toggle('Running _read_census_table sub-function', verbose=verbose)
            imports = []
            for census_table in census_tables:
                table = pd.read_csv(os.path.join(dat_path, census_table)).iloc[:, const.CENSUS_COL_NEEDED]
                table.columns = const.CENSUS_COL_NAMES
                imports.append(table)

            imports = pd.concat(imports, sort=True)  # combine into a single DataFrame
            imports = pd.wide_to_long(imports,
                                      stubnames='cpt',
                                      i='geography_code',
                                      j='census_property_type'
                                      ).reset_index().rename(columns={'cpt': table_type})
            self._logger.info('_read_census_table sub-function completed')
            gen.print_w_toggle('_read_census_table sub-function completed', verbose=verbose)
            return imports

        # Reads the population and property data
        population_imports = _read_census_table(population_tables, 'population')
        property_imports = _read_census_table(property_tables, 'properties')

        # Reads the geometry
        geography = gen.safe_read_csv(geography).iloc[:, 0:3]

        # Merges the population data, property data and geometry into a single DataFrame: household_occupancy
        household_occupancy = population_imports.merge(property_imports,
                                                       how='left',
                                                       on=['geography_code', 'census_property_type'])
        household_occupancy = geography.merge(household_occupancy,
                                              how='left',
                                              left_on='lsoa11cd',
                                              right_on='geography_code')

        # Calculate the household occupancy ratio
        household_occupancy['household_occupancy'] = household_occupancy['population'] / household_occupancy[
            'properties']
        self._logger.info('lsoa_census_data_prep function completed')
        gen.print_w_toggle('lsoa_census_data_prep function completed', verbose=verbose)
        return household_occupancy

    # Sub-function used by apply_household_occupancy. Not called directly by census_lu.py
    # TODO: improve the docstring here

    # Sub-sub-function used by apply_household_occupancy, called by balance_missing_hops.
    # Not called directly by census_lu.py

    def zone_up(self,
                cpt_data: pd.DataFrame,
                grouping_col: str,
                verbose: bool = True,
                ) -> pd.DataFrame:

        """
        Function to raise up a level of spatial aggregation & aggregate at that level, then bring new factors back down
        # TODO: Might be nice to have this zone up any level of zonal aggregation
        Raise LSOA to MSOA for spatial aggregation

        Parameters:
        -----------
        cpt_data: pd.DataFrame
            Dataframe containing census property type data.

        grouping_col: str


        verbose: bool = True
            Whether to print text or not.

        Returns:
        --------
        cpt_data:

        """
        self._logger.info('Running zone_up function')
        gen.print_w_toggle('Running zone_up function', verbose=verbose)

        zone_translation = self.read_zt_file()

        # Audit any missing objectids
        dat_lsoas = len(cpt_data['objectid'].unique())
        zt_lsoas = len(zone_translation['lsoaZoneID'].unique())

        if dat_lsoas == zt_lsoas:
            gen.print_w_toggle('zones match 1:1 - zoning up should be smooth', verbose=verbose)
        else:
            gen.print_w_toggle('some zones missing for LSOA-MSOA zone translation:', dat_lsoas - zt_lsoas,
                               verbose=verbose)

        cpt_data = cpt_data.rename(columns=const.CPT_DATA_COL)
        cpt_data = cpt_data.merge(zone_translation, how='left', on='lsoaZoneID').reset_index()
        cpt_data = self.aggregate_cpt(cpt_data, grouping_col=grouping_col)

        self._logger.info('zone_up function completed')
        gen.print_w_toggle('zone_up function completed', verbose=verbose)

        return cpt_data

    # Sub-sub-(sub)-function used by apply_household_occupancy, called by balance_missing_hops (and zone_up).
    # Not called directly by census_lu.py
    def aggregate_cpt(self,
                      cpt_data,
                      grouping_col=None,
                      verbose: bool = True,
                      ) -> pd.DataFrame:
        """
        Take some census property type data and return hops totals
        """
        self._logger.info('Running function aggregate_cpt')
        gen.print_w_toggle('Running function aggregate_cpt', verbose=verbose)
        if not grouping_col:
            cpt_data = cpt_data.loc[:, ['census_property_type', 'population', 'properties']]
            agg_data = cpt_data.groupby('census_property_type').sum().reset_index()
            agg_data['household_occupancy'] = agg_data['population'] / agg_data['properties']
        else:
            cpt_data = cpt_data.loc[:, ['census_property_type', 'population', 'properties', grouping_col]]
            agg_data = cpt_data.groupby(['census_property_type', grouping_col]).sum().reset_index()
            agg_data['household_occupancy'] = agg_data['population'] / agg_data['properties']
        self._logger.info('function aggregate_cpt complete')
        gen.print_w_toggle('function aggregate_cpt complete', verbose=verbose)
        return agg_data

    def balance_missing_hops(self,
                             cpt_data: pd.DataFrame,
                             grouping_col: str,
                             verbose: bool = True,
                             ) -> pd.DataFrame:
        """
        # TODO: Replace global with LAD or Country - likely to be marginal improvements. Currently UK-wide
        This function resolves the  msoa/lad household occupancy

        Parameters
        ----------
        cpt_data: pd.DataFrame
            Dataframe containing census property type data.

        grouping_col: str
            String containing the column name to be grouped.

        verbose: bool = True
            Whether to print text or not.

        Return
        ------
        cpt_data: pd.Dataframe

        """
        self._logger.info('Running function balanced_missing_hops')
        gen.print_w_toggle('Running function balanced_missing_hops', verbose=verbose)

        msoa_agg = self.zone_up(cpt_data=cpt_data, grouping_col=grouping_col)
        msoa_agg = msoa_agg.loc[:, [grouping_col, 'census_property_type',
                                    'household_occupancy']].rename(columns={'household_occupancy': 'msoa_ho'})

        global_agg = self.zone_up(cpt_data, grouping_col=grouping_col)

        global_agg = self.aggregate_cpt(global_agg, grouping_col=None)
        global_agg = global_agg.loc[:, ['census_property_type',
                                        'household_occupancy']].rename(columns={'household_occupancy': 'global_ho'})

        cpt_data = msoa_agg.merge(global_agg, how='left', on='census_property_type')

        gen.print_w_toggle('Resolving ambiguous household occupancies', verbose=verbose)

        # Use the global household occupancy where the MSOA household occupancy is unavailable
        cpt_data['household_occupancy'] = cpt_data['msoa_ho'].fillna(cpt_data['global_ho'])
        cpt_data['ho_type'] = np.where(np.isnan(cpt_data['msoa_ho']), 'global',
                                       'msoa')  # record where global has been used
        cpt_data = cpt_data.drop(['msoa_ho', 'global_ho'], axis=1)

        self._logger.info('Function balanced_missing_hops completed')
        gen.print_w_toggle('Function balanced_missing_hops completed', verbose=verbose)

        return cpt_data

    def apply_household_occupancy(self,
                                  prob_filled: pd.DataFrame,
                                  do_import: bool = False,
                                  write_out: bool = True,
                                  verbose: bool = True,
                                  ) -> pd.DataFrame:

        """
        Applies growth factor to estimate the zonal household population.

        Parameters
        ----------
        prob_filled: pd.Dataframe
            Dataframe containing probability of all properties occupied
            by zone.

        do_import:
            Whether to import census property type data or not.

        write_out:
            Whether to export the final output dataframe as csv or not.

        verbose: bool = True
            Whether to print text or not.

        Returns
        -------
        all_res_property:
            Dataframe containing estimated household population, number of
            properties and the household occupancy adjusted to account for
            the projection from the 2011 census year to the base year and
            the filled property adjustment by zone and dwelling type.

        """
        # TODO: want to be able to run at LSOA level when point correspondence is done.
        # TODO: Folders for outputs to separate this process from the household classification

        # TODO: Move the 2011 process step to the census lu object
        self._logger.info('Running Step 3.2.3')
        gen.print_w_toggle('Running Step 3.2.3', verbose=verbose)

        if do_import:
            balanced_cpt_data = gen.safe_read_csv(self.bal_cpt_data_path)
        else:
            cpt_data = self.lsoa_census_data_prep(
                dat_path=self.census_dat,
                population_tables=[const.EWQS401, const.SQS401],
                property_tables=[const.EWQS402, const.SQS402],
                geography=const.DEFAULT_LSOAREF,
            )

            # Zone up here to MSOA aggregations
            balanced_cpt_data = self.balance_missing_hops(cpt_data=cpt_data, grouping_col='msoaZoneID')
            balanced_cpt_data = balanced_cpt_data.fillna(0)

            balanced_cpt_data = balanced_cpt_data.merge(prob_filled, how='outer', on='msoaZoneID')
            balanced_cpt_data['household_occupancy'] = (balanced_cpt_data['household_occupancy'] *
                                                        balanced_cpt_data['Prob_DwellsFilled'])
            balanced_cpt_data = balanced_cpt_data.drop(columns={'Prob_DwellsFilled'})

            gen.safe_dataframe_to_csv(balanced_cpt_data, self.bal_cpt_data_path, index=False)

        # Visual spot checks - count zones, check cpt
        audit = balanced_cpt_data.groupby(['msoaZoneID']).count().reset_index()
        uk_msoa = gen.safe_read_csv(const.DEFAULT_MSOAREF, usecols=['objectid', 'msoa11cd'])
        gen.print_w_toggle('census hops zones = %d should be %d' % (audit['msoaZoneID'].drop_duplicates().count(),
                                                                    len(uk_msoa)), verbose=verbose)
        gen.print_w_toggle('counts of property type by zone', audit['census_property_type'].drop_duplicates(),
                           verbose=verbose)

        # Join MSOA ids to balanced cptdata
        uk_msoa = uk_msoa.rename(columns={'msoa11cd': 'msoaZoneID'})
        balanced_cpt_data = balanced_cpt_data.merge(uk_msoa, how='left', on='msoaZoneID').drop('objectid', axis=1)

        # Join MSOA to lad translation
        lad_translation = gen.safe_read_csv(const.ZONE_TRANSLATION_PATH_LAD_MSOA)
        lad_translation = lad_translation.rename(columns={'lad_zone_id': 'ladZoneID', 'msoa_zone_id': 'msoaZoneID'})
        lad_translation = lad_translation[['ladZoneID', 'msoaZoneID']]
        balanced_cpt_data = balanced_cpt_data.merge(lad_translation, how='left', on='msoaZoneID')

        # Join LAD code
        uk_lad = gen.safe_read_csv(const.DEFAULT_LADREF, usecols=['objectid', 'lad17cd'])
        balanced_cpt_data = balanced_cpt_data.merge(uk_lad, how='left', left_on='ladZoneID', right_on='objectid')

        # Check the join
        if len(balanced_cpt_data['ladZoneID'].unique()) == len(lad_translation['ladZoneID'].unique()):
            gen.print_w_toggle('All LADs joined properly', verbose=verbose)
            self._logger.info('All LADs joined properly')
        else:
            self._logger.info('Some LAD zones not accounted for')
            gen.print_w_toggle('Some LAD zones not accounted for', verbose=verbose)

        # Read in HOPS growth data
        balanced_cpt_data = balanced_cpt_data.drop(['ladZoneID', 'objectid'], axis=1)
        hops_growth = gen.safe_read_csv(self.hops_path)[['Area code', '11_to_%s' % self.base_year[-2:]]]

        # Uplift the figures to the Base Year
        # TODO work out if this is uplifting properly for years other than 2018
        balanced_cpt_data = balanced_cpt_data.merge(hops_growth,
                                                    how='left', left_on='lad17cd',
                                                    right_on='Area code').drop('Area code', axis=1).reset_index(
            drop=True)
        balanced_cpt_data[('household_occupancy_%s' % self.base_year[-2:])] = (
                balanced_cpt_data['household_occupancy'] *
                (1 + balanced_cpt_data[
                    ('11_to_%s' % self.base_year[-2:])]))
        trim_cols = ['msoaZoneID', 'census_property_type', 'household_occupancy_%s' % self.base_year[-2:], 'ho_type']
        balanced_cpt_data = balanced_cpt_data[trim_cols]

        # Read in all res property for the level of aggregation
        gen.print_w_toggle('Reading in AddressBase extract', verbose=verbose)

        all_res_property = pd.read_csv(self.addressbase_extract_path)[['ZoneID', 'census_property_type', 'UPRN']]
        all_res_property = all_res_property.groupby(['ZoneID', 'census_property_type']).count().reset_index()

        if self.model_zoning == 'MSOA':
            all_res_property = all_res_property.merge(balanced_cpt_data,
                                                      how='inner',
                                                      left_on=['ZoneID', 'census_property_type'],
                                                      right_on=['msoaZoneID', 'census_property_type'])
            all_res_property = all_res_property.drop('msoaZoneID', axis=1)

            # Audit join - ensure all zones accounted for
            if all_res_property['ZoneID'].drop_duplicates().count() != uk_msoa[
                'msoaZoneID'
            ].drop_duplicates().count():
                ValueError('Some zones dropped in Hops join')
            else:
                gen.print_w_toggle('All Hops areas accounted for', verbose=verbose)

            # allResPropertyZonal.merge(filled_properties, on = 'ZoneID')
            all_res_property['population'] = all_res_property['UPRN'] * all_res_property[
                'household_occupancy_%s' % self.base_year[-2:]]

            # audit
            audits.audit_3_2_3(self, all_res_property=all_res_property)

            if write_out:
                gen.safe_dataframe_to_csv(all_res_property, self.apply_hh_occ_path, index=False)

            self._logger.info('Step 3.2.3 completed')
            gen.print_w_toggle('Step 3.2.3 completed', verbose=verbose)
            return all_res_property

        else:
            self._logger.info("No support for this zoning system")
            gen.print_w_toggle(
                "No support for this zoning system", verbose=verbose)
            # only the MSOA zoning system is supported at the moment

    def property_type_refactor(self,
                               crp: pd.DataFrame,
                               verbose: bool = True,
                               ) -> pd.DataFrame:
        """
        Combines all flats into one category, i.e. property types = 4,5,6 and 7.

        Parameters
        ----------
        crp: pd.DataFrame
            Dataframe containing estimated household population in
            7 property types.

        verbose: bool = True
            Whether to print text or not.

        Return
        ------
        crp: pd.DataFrame
            Dataframe containing estimated household population combined to
            4 property types.
        """
        # TODO: This is a property type refactor, should be name like that
        self._logger.info('Running Step 3.2.4')
        gen.print_w_toggle('Running Step 3.2.4', verbose=verbose)
        crp = crp[self.crp_cols]
        crp_for_audit = crp.copy()
        # crp = crp.rename(columns={'census_property_type': 'property_type'})
        # Combine all flat types (4,5,6) and type 7.
        # Combine 4,5,6 and 7 dwelling types to 4.

        crp['census_property_type'] = crp['census_property_type'].map(const.PROPERTY_TYPE)
        crp['popXocc'] = crp['population'] * crp['household_occupancy_%s' % self.base_year[-2:]]
        crp = crp.groupby(const.CRP_COLS[:2]).sum().reset_index()
        crp['household_occupancy_%s' % self.base_year[-2:]] = crp['popXocc'] / crp[
            'population']  # compute the weighted average occupancy
        crp = crp.drop('popXocc', axis=1)
        self._logger.info('Population currently {}'.format(crp.population.sum()))
        processed_crp_for_audit = crp.copy()
        gen.safe_dataframe_to_csv(crp, self.lu_formatting_path, index=False)

        # Audit
        audits.audit_3_2_4(self, crp, crp_for_audit, processed_crp_for_audit)
        self._logger.info('Step 3.2.4 completed')
        gen.print_w_toggle('Step 3.2.4 completed', verbose=verbose)
        return crp

    def mye_aps_process(self,
                        called_by: str,
                        verbose: bool = True,
                        ):
        """
        Estimates adjusted NTEM base year population.

         Parameters
         ----------
         called_by: str
             String containing information from where it is called.
             Depending on the string the relevant outputs are processed.

         verbose: bool = True
             Whether to print text or not.

         Return
         ------
         mye_aps_process_output: pd.Dataframe or list
             Returns zonal household population or worker and non worker population by district,
             lad level adjustents and population estimate by district, age and gender depending
             on the function called.

         """

        self._logger.info('Running MYE_APS process function')
        self._logger.info('This has been called by %s' % called_by)
        gen.print_w_toggle('Running MYE_APS process function', verbose=verbose)
        gen.print_w_toggle('This has been called by %s' % called_by, verbose=verbose)

        # Read in files
        nomis_mype_msoa_age_gender = gen.safe_read_csv(self.nomis_mype_msoa_age_gender_path,
                                                       skiprows=6, skip_blank_lines=True)
        uk_2011_and_2021_la = gen.safe_read_csv(self.uk_2011_and_2021_la_path)
        scottish_2011_z2la = gen.safe_read_csv(self.scottish_2011_z2la_path)
        lookup_geography_2011 = gen.safe_read_csv(const.LOOKUP_GEOGRAPHY_2011_PATH)
        qs101_uk = gen.safe_read_csv(const.QS101_UK_PATH, skiprows=7).dropna()
        scottish_base_year_males = gen.safe_read_csv(self.scottish_base_year_males_path)
        scottish_base_year_females = gen.safe_read_csv(self.scottish_base_year_females_path)
        # la_to_msoa_uk = gen.safe_read_csv(
        #     os.path.join(la_to_msoa_directory, la_to_msoa_path_og))  # Original path, proportions don't quite add to 1
        # Path with manual corrections to make proportions equal 1
        la_to_msoa_uk = gen.safe_read_csv(const.LA_TO_MSOA_UK_PATH)
        scottish_la_changes_post_2011 = gen.safe_read_csv(self.scottish_la_changes_post_2011_path)
        aps_ftpt_gender_base_year = gen.safe_read_csv(self.aps_ftpt_gender_base_year_path,
                                                      skiprows=17, skip_blank_lines=True)
        nomis_base_year_mye_pop_by_la = gen.safe_read_csv(self.nomis_base_year_mye_pop_by_la_path,
                                                          skiprows=9, skip_blank_lines=True)
        aps_soc = gen.safe_read_csv(self.aps_soc_path, skiprows=12, skip_blank_lines=True)

        # Processing to fix data types or missing headers
        lookup_geography_2011_ew_only = lookup_geography_2011.dropna().copy()  # Removes Scotland and the nans
        lookup_geography_2011_ew_only['Grouped LA'] = lookup_geography_2011_ew_only['Grouped LA'].astype(int)
        qs101_uk.rename(columns={'Unnamed: 1': 'MSOA'}, inplace=True)
        nomis_base_year_mye_pop_by_la.rename(columns={'Unnamed: 1': '2021_LA'}, inplace=True)

        def process_age_based_pop(padp_df: pd.DataFrame) -> pd.DataFrame:
            # Now begin manipulation
            """
            Group population in dataframe based on required age groups.

            Parameters
            ----------
            padp_df:
                Dataframe containing population by age.

            Returns
            -------
            padp_df:
                Dataframe containing population grouped by required age groups.
            """

            padp_df['16-74'] = (padp_df['Aged 16 to 64'].astype(int) +
                                padp_df['Aged 65-69'].astype(int) +
                                padp_df['Aged 70-74'].astype(int))
            padp_df['under_16'] = padp_df['Aged 0 to 15'].astype(int)
            padp_df['75_and_over'] = (padp_df['All Ages'].astype(int) -
                                      (padp_df['16-74'] + padp_df['under_16'].astype(int)))
            padp_df = padp_df[['mnemonic', 'under_16', '16-74', '75_and_over']]
            return padp_df

        # Create the zone to LA lookup for the all_residents df
        all_residents_geography = lookup_geography_2011_ew_only.copy()
        all_residents_geography = all_residents_geography[['NorMITs Zone', '2011 LA', 'MSOA']]
        all_residents_geography.rename(columns={'NorMITs Zone': 'Zone', '2011 LA': '2011_LA'}, inplace=True)

        # Format the scottish zones geography for easy read from qs101
        scottish_2011_z2la_for_qs101 = scottish_2011_z2la.copy()
        scottish_2011_z2la_for_qs101 = scottish_2011_z2la_for_qs101[['MSOA', 'NorMITs Zone', 'LA Code']]
        scottish_2011_z2la_for_qs101.rename(columns={'NorMITs Zone': 'Zone', 'LA Code': '2011_LA'}, inplace=True)

        # Now need to format E&W 2011 geography in the same way...
        englandwales_2011_z2la_for_qs101 = all_residents_geography.copy()
        englandwales_2011_z2la_for_qs101 = englandwales_2011_z2la_for_qs101[['MSOA', 'Zone', '2011_LA']]

        # Append rows from scotland onto the end of the E&W df
        # Also check max zone = max index + 1 (+1 due to 0 indexing)
        uk_2011_z2la_for_qs101 = englandwales_2011_z2la_for_qs101.append(scottish_2011_z2la_for_qs101).reset_index()
        uk_2011_z2la_for_qs101.drop(columns=['index'], inplace=True)
        max_z_uk_2011_z2la_for_qs101 = uk_2011_z2la_for_qs101['Zone'].max()
        max_i_uk_2011_z2la_for_qs101 = uk_2011_z2la_for_qs101.shape
        max_i_uk_2011_z2la_for_qs101 = max_i_uk_2011_z2la_for_qs101[0]
        if max_z_uk_2011_z2la_for_qs101 == max_i_uk_2011_z2la_for_qs101:
            self._logger.info('All %s zones accounted for in the UK' % str(max_z_uk_2011_z2la_for_qs101))
        else:
            self._logger.warning('!!!!! WARNING !!!!!')
            self._logger.warning('Something is wrong with the UK zonal data')
            self._logger.warning('Expected %s zones' % str(max_z_uk_2011_z2la_for_qs101))
            self._logger.warning('Got %s zones' % str(max_i_uk_2011_z2la_for_qs101))
            gen.print_w_toggle('!!!!! WARNING !!!!!', verbose=verbose)
            gen.print_w_toggle('Something is wrong with the UK zonal data', verbose=verbose)
            gen.print_w_toggle('Expected', max_z_uk_2011_z2la_for_qs101, 'zones', verbose=verbose)
            gen.print_w_toggle('Got', max_i_uk_2011_z2la_for_qs101, 'zones', verbose=verbose)

        # Process UK 2021 geography into a format for QS101 processing
        qs101_uk_2011_and_2021_la = uk_2011_and_2021_la.copy()
        qs101_uk_2011_and_2021_la = qs101_uk_2011_and_2021_la[['2011 LA Code', '2021 LA Code']]
        qs101_uk_2011_and_2021_la.rename(columns={'2011 LA Code': '2011_LA', '2021 LA Code': '2021_LA'}, inplace=True)

        # Cut scottish_la_changes_post_2011 down to a useful lookup
        recoded_scottish_lads_post_2011 = scottish_la_changes_post_2011.copy()
        recoded_scottish_lads_post_2011 = recoded_scottish_lads_post_2011[['CA', 'CAName', 'CADateEnacted']]
        recoded_scottish_lads_post_2011 = recoded_scottish_lads_post_2011.loc[
            recoded_scottish_lads_post_2011['CADateEnacted'] >= 20111231]  # New codes post 12th Dec 2011
        recoded_scottish_lads_post_2011.reset_index(inplace=True)
        recoded_scottish_lads_post_2011.drop(columns=['index', 'CADateEnacted'], inplace=True)
        recoded_scottish_lads_post_2011.rename(columns={'CA': 'New area code', 'CAName': 'Area1'}, inplace=True)

        # Process Scottish male and female data into a format for all_residents processing
        lad_scottish_base_year_males = scottish_base_year_males.copy()
        lad_scottish_base_year_males = pd.merge(lad_scottish_base_year_males,
                                                recoded_scottish_lads_post_2011,
                                                on='Area1',
                                                how='left')
        lad_scottish_base_year_males['New area code'] = lad_scottish_base_year_males['New area code'].fillna(0)
        lad_scottish_base_year_males['Area code'] = np.where(lad_scottish_base_year_males['New area code'] == 0,
                                                             lad_scottish_base_year_males['Area code'],
                                                             lad_scottish_base_year_males['New area code'])
        lad_scottish_base_year_males.drop(columns=['New area code'], inplace=True)
        lad_scottish_base_year_males['M_Total'] = lad_scottish_base_year_males.iloc[:, 3:].sum(axis=1)
        lad_scottish_base_year_males = lad_scottish_base_year_males[
            ['Area code', 'M_Total', 'under 16', '16-74', '75 or over']]
        lad_scottish_base_year_males.rename(columns={'Area code': '2011_LA',
                                                     'under 16': 'M_under_16',
                                                     '16-74': 'M_16-74',
                                                     '75 or over': 'M_75_and_over'}, inplace=True)
        lad_scottish_base_year_females = scottish_base_year_females.copy()
        lad_scottish_base_year_females = pd.merge(lad_scottish_base_year_females,
                                                  recoded_scottish_lads_post_2011,
                                                  on='Area1',
                                                  how='left')
        lad_scottish_base_year_females['New area code'] = lad_scottish_base_year_females['New area code'].fillna(0)
        lad_scottish_base_year_females['Area code'] = np.where(lad_scottish_base_year_females['New area code'] == 0,
                                                               lad_scottish_base_year_females['Area code'],
                                                               lad_scottish_base_year_females['New area code'])
        lad_scottish_base_year_females.drop(columns=['New area code'], inplace=True)
        lad_scottish_base_year_females['F_Total'] = lad_scottish_base_year_females.iloc[:, 3:].sum(axis=1)
        lad_scottish_base_year_females = lad_scottish_base_year_females[
            ['Area code', 'F_Total', 'under 16', '16-74', '75 or over']]
        lad_scottish_base_year_females.rename(columns={'Area code': '2011_LA',
                                                       'under 16': 'F_under_16',
                                                       '16-74': 'F_16-74',
                                                       '75 or over': 'F_75_and_over'}, inplace=True)
        lad_scottish_base_year_all_pop = pd.merge(lad_scottish_base_year_males,
                                                  lad_scottish_base_year_females,
                                                  how='outer',
                                                  on='2011_LA')
        lad_scottish_base_year_all_pop['M_Total'] = (lad_scottish_base_year_all_pop['M_Total'] +
                                                     lad_scottish_base_year_all_pop['F_Total'])
        lad_scottish_base_year_all_pop.rename(columns={'M_Total': 'Total'}, inplace=True)
        lad_scottish_base_year_all_pop.drop(columns=['F_Total'], inplace=True)
        # Now check that the total is still valid
        scottish_male_total = scottish_base_year_males.iloc[:, 3:].sum(axis=1)
        scottish_female_total = scottish_base_year_females.iloc[:, 3:].sum(axis=1)
        scottish_pop_total = scottish_male_total.sum() + scottish_female_total.sum()
        if scottish_pop_total == lad_scottish_base_year_all_pop['Total'].sum():
            self._logger.info('All %s people accounted for in Scotland', str(scottish_pop_total))
        else:
            self._logger.warning('!!!!! WARNING !!!!!')
            self._logger.warning('Something is wrong with the Scottish population data')
            self._logger.warning('Expected population of ' + scottish_pop_total)
            self._logger.warning('Got a population of ' + lad_scottish_base_year_all_pop['Total'].sum())
            gen.print_w_toggle('!!!!! WARNING !!!!!', verbose=verbose)
            gen.print_w_toggle('Something is wrong with the Scottish population data', verbose=verbose)
            gen.print_w_toggle('Expected population of', scottish_pop_total, verbose=verbose)
            gen.print_w_toggle('Got a population of', lad_scottish_base_year_all_pop['Total'].sum(), verbose=verbose)

        # Format lookup for LA to MSOA
        la_to_msoa_uk_lookup = la_to_msoa_uk.copy()
        la_to_msoa_uk_lookup = la_to_msoa_uk_lookup[['msoa_zone_id', 'lad_to_msoa']]
        la_to_msoa_uk_lookup.rename(columns={'msoa_zone_id': 'MSOA'}, inplace=True)
        la_to_msoa_uk_lookup['lad_to_msoa'].sum()

        # Strip Northern Ireland from the Nomis Base Year MYPE
        nomis_base_year_mye_pop_by_la_gb = nomis_base_year_mye_pop_by_la.copy()
        nomis_base_year_mye_pop_by_la_gb = nomis_base_year_mye_pop_by_la_gb[
            ~nomis_base_year_mye_pop_by_la_gb['2021_LA'].str.contains('N')]

        # Process APS data

        # Remove the Isles of Scilly with UK average data (as it's missing in APS data)
        # Also strip the totals row
        aps_ftpt_gender_base_year_to_use = aps_ftpt_gender_base_year.copy()
        aps_ftpt_gender_base_year_to_use.dropna(inplace=True)  # Drops total column
        # Following lines not required unless you wish to interrogate some of the checking dfs in more detail
        # aps_ftpt_gender_base_year_to_use_la_list = aps_ftpt_gender_base_year_to_use.copy()
        # aps_ftpt_gender_base_year_to_use_la_list = aps_ftpt_gender_base_year_to_use_la_list[['LAD']]

        # Deal with Scilly
        aps_ftpt_gender_base_year_to_use = aps_ftpt_gender_base_year_to_use.set_index(['LAD'])
        aps_ftpt_gender_base_year_scilly_pulled = aps_ftpt_gender_base_year_to_use.loc[
            ['Isles of Scilly']].reset_index()
        aps_ftpt_gender_base_year_to_use.drop(['Isles of Scilly'], inplace=True)
        aps_ftpt_gender_base_year_to_use.reset_index(inplace=True)

        aps_ftpt_gender_base_year_uk_ave_cols = list(aps_ftpt_gender_base_year_to_use.columns)
        aps_ftpt_gender_base_year_uk_ave_cols = aps_ftpt_gender_base_year_uk_ave_cols[2:]
        aps_ftpt_gender_base_year_to_use.replace(',', '', regex=True, inplace=True)
        # # indicates data that is deemed 'statistically unreliable' by ONS.
        # It occurs in only 2 locations, mostly in the south in the columns
        # we are interested in here. Whilst a more robust approach might be
        # to estimate a figure to replace it, it is being set to 0 here for
        # simplicity, as any other approach would require bespoke solutions
        # for each instance.
        aps_ftpt_gender_base_year_to_use.replace('#', '0', regex=True, inplace=True)
        # ! indicate a small (0-2) sample size. Setting values to 0 where this occurs
        aps_ftpt_gender_base_year_to_use.replace('!', '0', regex=True, inplace=True)
        # Ditto for *, but sample size is in range 3-9. Setting values to 0 here too...
        aps_ftpt_gender_base_year_to_use.replace('*', '0', inplace=True)
        # - indicates missing data. Only appears in confidence intervals columns
        # and the (removed) Isles of Scilly row, so replacing with 0 for simplicity
        # Also, all numbers in this dataset should be +ve, so no risk of removing
        # -ve values!
        aps_ftpt_gender_base_year_to_use.replace('-', '0', regex=True, inplace=True)
        # ~ indicates an absolute value <500. It occurs only in the 'Males
        # part time employment' column for the Outer Hebrides. As the Outer
        # Hebrides are not famed for having a large part time workforce,
        # setting this to 0 too.
        aps_ftpt_gender_base_year_to_use.replace('~', '0', inplace=True)
        aps_ftpt_gender_base_year_to_use[aps_ftpt_gender_base_year_uk_ave_cols] = aps_ftpt_gender_base_year_to_use[
            aps_ftpt_gender_base_year_uk_ave_cols].astype(float)
        # Re-instate all the examples of '-' LAD in names
        aps_ftpt_gender_base_year_to_use['LAD'].replace('0', '-', regex=True, inplace=True)

        # Process APS workers data
        aps_ftpt_gender_base_year_summary = aps_ftpt_gender_base_year_to_use.copy()
        aps_ftpt_gender_base_year_summary_cols = [
            col for col in aps_ftpt_gender_base_year_summary.columns if 'numerator' in col]
        aps_ftpt_gender_base_year_summary = aps_ftpt_gender_base_year_summary[aps_ftpt_gender_base_year_summary_cols]
        aps_ftpt_gender_base_year_summary_cols2 = [
            col for col in aps_ftpt_gender_base_year_summary.columns if 'male' in col]
        aps_ftpt_gender_base_year_summary = aps_ftpt_gender_base_year_summary[aps_ftpt_gender_base_year_summary_cols2]
        aps_ftpt_gender_base_year_summary_cols3 = [s.replace(' - aged 16-64 numerator', '') for s in
                                                   aps_ftpt_gender_base_year_summary_cols2]
        aps_ftpt_gender_base_year_summary_cols3 = [s.replace('s in employment working', '') for s in
                                                   aps_ftpt_gender_base_year_summary_cols3]
        aps_ftpt_gender_base_year_summary_cols3 = [
            s.replace('% of ', '') for s in aps_ftpt_gender_base_year_summary_cols3]
        aps_ftpt_gender_base_year_summary_cols3 = [
            s.replace('full-time', 'fte') for s in aps_ftpt_gender_base_year_summary_cols3]
        aps_ftpt_gender_base_year_summary_cols3 = [
            s.replace('part-time', 'pte') for s in aps_ftpt_gender_base_year_summary_cols3]
        aps_ftpt_gender_base_year_summary.columns = aps_ftpt_gender_base_year_summary_cols3
        aps_ftpt_gender_base_year_summary['Total Worker 16-64'] = aps_ftpt_gender_base_year_summary.sum(axis=1)

        aps_ftpt_gender_base_year_rows = aps_ftpt_gender_base_year_to_use.copy()
        aps_ftpt_gender_base_year_rows = aps_ftpt_gender_base_year_rows.iloc[:, :2]
        aps_ftpt_gender_base_year_summary = aps_ftpt_gender_base_year_rows.join(aps_ftpt_gender_base_year_summary)

        aps_ftpt_gender_base_year_summary_percent = aps_ftpt_gender_base_year_summary.copy()
        aps_ftpt_gender_base_year_summary_percent = aps_ftpt_gender_base_year_summary_percent[
            aps_ftpt_gender_base_year_summary_cols3].divide(
            aps_ftpt_gender_base_year_summary_percent['Total Worker 16-64'], axis='index')

        # Actually might want to do this Scilly bit right at the very end once merged with the rest of the data?
        aps_ftpt_gender_base_year_scilly = aps_ftpt_gender_base_year_summary_percent.mean()
        aps_ftpt_gender_base_year_scilly = pd.DataFrame(aps_ftpt_gender_base_year_scilly)
        aps_ftpt_gender_base_year_scilly = aps_ftpt_gender_base_year_scilly.transpose()
        aps_ftpt_gender_base_year_scilly['Checksum'] = aps_ftpt_gender_base_year_scilly.sum(axis=1)
        aps_ftpt_gender_base_year_scilly['Checksum'] = aps_ftpt_gender_base_year_scilly['Checksum'] - 1
        scilly_rows = aps_ftpt_gender_base_year_scilly_pulled.copy()
        scilly_rows = scilly_rows.iloc[:, :2]
        scilly_rows = scilly_rows.join(aps_ftpt_gender_base_year_scilly)

        aps_ftpt_gender_base_year_summary_percent = aps_ftpt_gender_base_year_rows.join(
            aps_ftpt_gender_base_year_summary_percent)
        aps_ftpt_gender_base_year_summary_percent['Checksum'] = aps_ftpt_gender_base_year_summary_percent.iloc[
                                                                :, -4:].sum(axis=1)
        aps_ftpt_gender_base_year_summary_percent['Checksum'] = aps_ftpt_gender_base_year_summary_percent[
                                                                    'Checksum'] - 1
        aps_ftpt_gender_base_year_summary_percent = aps_ftpt_gender_base_year_summary_percent.append(scilly_rows)
        if abs(aps_ftpt_gender_base_year_summary_percent['Checksum'].sum()) < 0.000000001:
            self._logger.info('Sum of gender %ages across categories is close enough to 1 for all rows')
        else:
            self._logger.warning('!!!!! WARNING !!!!!')
            self._logger.warning('Summing across fte/pte and gender caused an error')
            self._logger.warning('All rows did not sum to 1')
            gen.print_w_toggle('!!!!! WARNING !!!!!', verbose=verbose)
            gen.print_w_toggle('Summing across fte/pte and gender caused an error', verbose=verbose)
            gen.print_w_toggle('All rows did not sum to 1', verbose=verbose)

        # Following lines would tidy up aps_ftpt_gender_base_year_summary_percent if you ever want to look at it
        # Also remember to un-comment the small section above the gives one of the variables
        # aps_ftpt_gender_base_year_summary_percent = aps_ftpt_gender_base_year_summary_percent.set_index('LAD')
        # aps_ftpt_gender_base_year_summary_percent = aps_ftpt_gender_base_year_summary_percent.reindex(
        #     index=aps_ftpt_gender_base_year_to_use_la_list['LAD'])
        # aps_ftpt_gender_base_year_summary_percent = aps_ftpt_gender_base_year_summary_percent.reset_index()

        # Repeat process for SOC data
        aps_soc_to_use = aps_soc.copy()
        aps_soc_to_use = aps_soc_to_use.set_index(['LAD'])
        aps_soc_to_use.drop(['Isles of Scilly', 'Column Total'], inplace=True)
        aps_soc_to_use.reset_index(inplace=True)
        aps_soc_to_use = aps_soc_to_use[
            aps_soc_to_use.columns.drop(list(aps_soc_to_use.filter(regex='Unemployment rate')))]
        aps_soc_to_use = aps_soc_to_use[
            aps_soc_to_use.columns.drop(list(aps_soc_to_use.filter(regex='percent')))]
        aps_soc_to_use = aps_soc_to_use[
            aps_soc_to_use.columns.drop(list(aps_soc_to_use.filter(regex='conf')))]
        aps_soc_to_use.replace(',', '', regex=True, inplace=True)
        # ! indicate a small (0-2) sample size. Setting values to 0 where this occurs
        aps_soc_to_use.replace('!', '0', regex=True, inplace=True)
        # Ditto for *, but sample size is in range 3-9. Setting values to 0 here too...
        aps_soc_to_use.replace('*', '0', inplace=True)
        # ~ indicates an absolute value <500. It occurs only in the 'Sales
        # customer service' column for the Orkney Islands. As the Orkney's
        # are not a haven for the call centre industry, setting this to 0
        # too.
        aps_soc_to_use.replace('~', '0', inplace=True)
        # # indicates data that is deemed 'statistically unreliable' by ONS.
        # It occurs in only 4 locations, mostly in the south in the columns
        # we are interested in here. Whilst a more robust approach might be
        # to estimate a figure to replace it, it is being set to 0 here for
        # simplicity, as any other approach would require bespoke solutions
        # for each instance.
        aps_soc_to_use.replace('#', '0', regex=True, inplace=True)
        aps_soc_to_use_ave_cols = list(aps_soc_to_use.columns)
        aps_soc_to_use_ave_cols = aps_soc_to_use_ave_cols[1:]
        aps_soc_to_use[aps_soc_to_use_ave_cols] = aps_soc_to_use[
            aps_soc_to_use_ave_cols].astype(float)

        aps_soc_to_use = aps_soc_to_use.rename(
            columns={'% of all people aged 16+ who are male denominator': 'Aged_16+'})
        aps_soc_to_use = aps_soc_to_use[
            aps_soc_to_use.columns.drop(list(aps_soc_to_use.filter(regex='denominator')))]
        aps_soc_to_use = aps_soc_to_use[
            aps_soc_to_use.columns.drop(list(aps_soc_to_use.filter(regex='male numerator')))]
        aps_soc_to_use_cols = list(aps_soc_to_use.columns)
        aps_soc_to_use_cols = [s.replace('% all in employment who are - ', '') for s in aps_soc_to_use_cols]
        aps_soc_to_use_cols_new = []
        aps_soc_to_use_cols_soc = []
        for s in aps_soc_to_use_cols:
            split_s = s.split(':', 1)[0]
            if len(s) > len(split_s):
                string_s = ['SOC', s.split(':', 1)[0]]
                aps_soc_to_use_cols_new.append(''.join(string_s))
                aps_soc_to_use_cols_soc.append(''.join(string_s))
            else:
                aps_soc_to_use_cols_new.append(s)
        aps_soc_to_use.set_axis(aps_soc_to_use_cols_new, axis=1, inplace=True)
        aps_soc_to_use['Total_Workers'] = aps_soc_to_use[aps_soc_to_use_cols_soc].sum(axis=1)

        # Turn SOC data into proportions by 2021 LA
        aps_soc_props = aps_soc_to_use.copy()
        aps_soc_props['higher'] = aps_soc_props['SOC1'] + aps_soc_props['SOC2'] + aps_soc_props['SOC3']
        aps_soc_props['medium'] = aps_soc_props['SOC4'] + aps_soc_props['SOC5'] + aps_soc_props['SOC6'] + aps_soc_props[
            'SOC7']
        aps_soc_props['skilled'] = aps_soc_props['SOC8'] + aps_soc_props['SOC9']
        aps_soc_props = aps_soc_props[
            aps_soc_props.columns.drop(list(aps_soc_props.filter(regex='SOC')))]
        aps_soc_props.drop(columns=['Aged_16+'], inplace=True)
        aps_soc_props = aps_soc_props.append(
            aps_soc_props.sum(numeric_only=True), ignore_index=True)
        aps_soc_props['LAD'].fillna("UK wide total", inplace=True)
        aps_soc_props['higher'] = aps_soc_props['higher'] / aps_soc_props['Total_Workers']
        aps_soc_props['medium'] = aps_soc_props['medium'] / aps_soc_props['Total_Workers']
        aps_soc_props['skilled'] = aps_soc_props['skilled'] / aps_soc_props['Total_Workers']
        aps_soc_props['Checksum'] = aps_soc_props['higher'] + aps_soc_props['medium'] + aps_soc_props['skilled'] - 1
        if abs(max(aps_soc_props['Checksum'])) < 0.000001 and abs(min(aps_soc_props['Checksum'])) < 0.000001:
            self._logger.info('All SOC proportions summed to 1')
            self._logger.info('(within reasonable deviation)')
            self._logger.info('Max deviation value was: %s' % str(max(aps_soc_props['Checksum'])))
            self._logger.info('Min deviation value was: %s' % str(min(aps_soc_props['Checksum'])))
        else:
            self._logger.warning('!!!!! WARNING !!!!!')
            self._logger.warning('SOC proportions did not sum to 1')
            self._logger.warning('(within reasonable deviation)')
            self._logger.warning('Max deviation value was: %s' % str(max(aps_soc_props['Checksum'])))
            self._logger.warning('Min deviation value was: %s' % str(min(aps_soc_props['Checksum'])))
            gen.print_w_toggle('!!!!! WARNING !!!!!', verbose=verbose)
            gen.print_w_toggle('SOC proportions did not sum to 1', verbose=verbose)
            gen.print_w_toggle('(within reasonable deviation)', verbose=verbose)
            gen.print_w_toggle('Max deviation value was:', max(aps_soc_props['Checksum']), verbose=verbose)
            gen.print_w_toggle('Min deviation value was:', min(aps_soc_props['Checksum']), verbose=verbose)
        aps_soc_props_to_add = aps_soc_props['LAD'] == 'UK wide total'
        aps_soc_props_to_add = aps_soc_props[aps_soc_props_to_add]
        aps_soc_props_to_add = aps_soc_props_to_add.replace('UK wide total', 'Isles of Scilly')
        aps_soc_props = aps_soc_props.append([aps_soc_props_to_add], ignore_index=True)
        aps_soc_props_to_merge = aps_soc_props.copy()
        aps_soc_props_to_merge = aps_soc_props_to_merge.drop(columns=['Total_Workers', 'Checksum'])

        # Turn gender/ftpt employment data into proportions by 2021 LA
        aps_ftpt_gender_base_year_props = aps_ftpt_gender_base_year_summary.copy()
        aps_ftpt_gender_base_year_props = aps_ftpt_gender_base_year_props.append(
            aps_ftpt_gender_base_year_props.sum(numeric_only=True), ignore_index=True)
        aps_ftpt_gender_base_year_props['LAD'].fillna("UK wide total", inplace=True)
        aps_ftpt_gender_base_year_props['2021_LA'].fillna("All_UK001", inplace=True)
        aps_ftpt_gender_base_year_props['male fte'] = (aps_ftpt_gender_base_year_props['male fte'] /
                                                       aps_ftpt_gender_base_year_props['Total Worker 16-64'])
        aps_ftpt_gender_base_year_props['male pte'] = (aps_ftpt_gender_base_year_props['male pte'] /
                                                       aps_ftpt_gender_base_year_props['Total Worker 16-64'])
        aps_ftpt_gender_base_year_props['female fte'] = (aps_ftpt_gender_base_year_props['female fte'] /
                                                         aps_ftpt_gender_base_year_props['Total Worker 16-64'])
        aps_ftpt_gender_base_year_props['female pte'] = (aps_ftpt_gender_base_year_props['female pte'] /
                                                         aps_ftpt_gender_base_year_props['Total Worker 16-64'])
        aps_ftpt_gender_base_year_props['Checksum'] = (aps_ftpt_gender_base_year_props['male fte']
                                                       + aps_ftpt_gender_base_year_props['male pte']
                                                       + aps_ftpt_gender_base_year_props['female fte']
                                                       + aps_ftpt_gender_base_year_props['female pte']
                                                       - 1)
        if (abs(max(aps_ftpt_gender_base_year_props['Checksum'])) < 0.000001 and
                abs(min(aps_ftpt_gender_base_year_props['Checksum'])) < 0.000001):
            self._logger.info('All ft/pt gender proportions summed to 1')
            self._logger.info('(within reasonable deviation)')
            self._logger.info('Max deviation value was: %s' % str(max(aps_ftpt_gender_base_year_props['Checksum'])))
            self._logger.info('Min deviation value was: %s' % str(min(aps_ftpt_gender_base_year_props['Checksum'])))
        else:
            self._logger.warning('!!!!! WARNING !!!!!')
            self._logger.warning('ft/pt gender proportions did not sum to 1')
            self._logger.warning('(within reasonable deviation)')
            self._logger.warning('Max deviation value was: %s' % str(max(aps_ftpt_gender_base_year_props['Checksum'])))
            self._logger.warning('Min deviation value was: %s' % str(min(aps_ftpt_gender_base_year_props['Checksum'])))
            gen.print_w_toggle('!!!!! WARNING !!!!!', verbose=verbose)
            gen.print_w_toggle('ft/pt gender proportions did not sum to 1', verbose=verbose)
            gen.print_w_toggle('(within reasonable deviation)', verbose=verbose)
            gen.print_w_toggle('Max deviation value was:', max(aps_ftpt_gender_base_year_props['Checksum']),
                               verbose=verbose)
            gen.print_w_toggle('Min deviation value was:', min(aps_ftpt_gender_base_year_props['Checksum']), verbose=verbose)
        aps_ftpt_gender_to_add = aps_ftpt_gender_base_year_props['LAD'] == 'UK wide total'
        aps_ftpt_gender_to_add = aps_ftpt_gender_base_year_props[aps_ftpt_gender_to_add]
        aps_ftpt_gender_to_add = aps_ftpt_gender_to_add.replace('UK wide total', 'Isles of Scilly')
        aps_ftpt_gender_to_add = aps_ftpt_gender_to_add.replace('All_UK001', 'E06000053')
        aps_ftpt_gender_base_year_props = aps_ftpt_gender_base_year_props.append(
            [aps_ftpt_gender_to_add], ignore_index=True)
        aps_ftpt_gender_base_year_props_to_merge = aps_ftpt_gender_base_year_props.copy()
        aps_ftpt_gender_base_year_props_to_merge.drop(columns=['Total Worker 16-64', 'Checksum'], inplace=True)

        # Merge SOC and ft/pt gender tables into a single APS props table
        aps_props_to_merge = pd.merge(aps_soc_props_to_merge,
                                      aps_ftpt_gender_base_year_props_to_merge,
                                      how='outer',
                                      on='LAD')
        aps_props_to_merge.drop(columns=['LAD'], inplace=True)

        # Create HHR from QS101UK data
        qs101_uk_by_z = qs101_uk.copy()
        qs101_uk_by_z = pd.merge(qs101_uk_by_z, uk_2011_z2la_for_qs101, how='outer', on='MSOA')
        qs101_uk_by_z = pd.merge(qs101_uk_by_z, qs101_uk_2011_and_2021_la, how='left', on='2011_LA')
        qs101_uk_by_z['All categories: Residence type'] = qs101_uk_by_z['All categories: Residence type'].str.replace(
            ',',
            '')
        qs101_uk_by_z['Lives in a household'] = qs101_uk_by_z['Lives in a household'].str.replace(',', '')
        qs101_uk_by_z['%_of_HHR'] = (qs101_uk_by_z['Lives in a household'].astype(int) /
                                     qs101_uk_by_z['All categories: Residence type'].astype(int))

        # Perform initial processing on the NOMIS data.
        # Manipulates the (rather badly) formatted csv to present male and female data in same df.
        # And formats the age ranges to match those used in NorMITs.
        nomis_all_residents = nomis_mype_msoa_age_gender.copy()
        header_rows = nomis_all_residents.loc[
            nomis_all_residents['2011 super output area - middle layer'] == '2011 super output area - middle layer']
        header_rows = header_rows.index.tolist()
        nomis_all_residents_total_genders = nomis_all_residents.iloc[:header_rows[0], :].reset_index()
        nomis_all_residents_male = nomis_all_residents.iloc[header_rows[0] + 1:header_rows[1], :].reset_index()
        nomis_all_residents_female = nomis_all_residents.iloc[header_rows[1] + 1:, :].reset_index()
        nomis_gender_indices = pd.DataFrame([nomis_all_residents_total_genders.index.max(),
                                             nomis_all_residents_male.index.max(),
                                             nomis_all_residents_female.index.max()])
        nomis_gender_indices_min = nomis_gender_indices[0].min() + 1
        nomis_all_residents_total_genders = nomis_all_residents_total_genders.iloc[:nomis_gender_indices_min, :].drop(
            columns=['index'])
        nomis_all_residents_total_check = nomis_all_residents_total_genders[['mnemonic', 'All Ages']]
        nomis_all_residents_male = nomis_all_residents_male.iloc[:nomis_gender_indices_min, :].drop(columns=['index'])
        nomis_all_residents_female = nomis_all_residents_female.iloc[:nomis_gender_indices_min, :].drop(
            columns=['index'])

        nomis_all_residents_male = process_age_based_pop(nomis_all_residents_male)
        nomis_all_residents_female = process_age_based_pop(nomis_all_residents_female)
        all_residents_headers = list(nomis_all_residents_male.columns)
        all_residents_male_headers = ['M_' + s for s in all_residents_headers]
        all_residents_female_headers = ['F_' + s for s in all_residents_headers]
        nomis_all_residents_male.columns = nomis_all_residents_male.columns[:-3].tolist() + all_residents_male_headers[
                                                                                            1:]
        nomis_all_residents_female.columns = nomis_all_residents_female.columns[
                                             :-3].tolist() + all_residents_female_headers[1:]

        all_residents = pd.merge(nomis_all_residents_male, nomis_all_residents_female, how='outer', on='mnemonic')
        all_residents = pd.merge(nomis_all_residents_total_check, all_residents, how='outer', on='mnemonic')

        # Check the male and female data sum to the total from source by MSOA.
        all_residents['All_Ages'] = all_residents.iloc[:, 2:].sum(axis=1)
        all_residents['All_Ages_Check'] = np.where(all_residents['All_Ages'] == all_residents['All Ages'].astype(int),
                                                   0, 1)
        if all_residents['All_Ages_Check'].sum() == 0:
            self._logger.info('Male and female totals summed across age bands')
            self._logger.info('Result successfully matched the input totals from NOMIS')
        else:
            self._logger.warning('!!!!! WARNING !!!!!!')
            self._logger.warning('Something went wrong when I tried to sum male and female')
            self._logger.warning('population totals across all age bands')
            self._logger.warning('I expected that \'All Ages\' should match\'All_Ages\',')
            self._logger.warning('but it did not in %s cases' % str(all_residents['All_Ages_Check'].sum()))
            self._logger.warning('The erroneous lines are:')
            self._logger.warning(all_residents.loc[all_residents['All_Ages_Check'] == 1])
            gen.print_w_toggle('!!!!! WARNING !!!!!!', verbose=verbose)
            gen.print_w_toggle('Something went wrong when I tried to sum male and female', verbose=verbose)
            gen.print_w_toggle('population totals across all age bands', verbose=verbose)
            gen.print_w_toggle('I expected that \'All Ages\' should match\'All_Ages\',', verbose=verbose)
            gen.print_w_toggle('but it did not in', all_residents['All_Ages_Check'].sum(), 'cases', verbose=verbose)
            gen.print_w_toggle('The erroneous lines are:', verbose=verbose)
            gen.print_w_toggle(all_residents.loc[all_residents['All_Ages_Check'] == 1], verbose=verbose)
        all_residents.drop(columns=['All_Ages', 'All_Ages_Check'], inplace=True)
        all_residents.rename(columns={'All Ages': 'Total', 'mnemonic': 'MSOA'}, inplace=True)

        all_residents = pd.merge(all_residents_geography, all_residents, how='outer', on='MSOA')

        # Create estimates of all_residents by zone in Scotland
        # As data is not available by zone in Scotland, the LA level data needs breaking down
        # An LA to MSOA conversion factor table is available
        all_residents_scotland = scottish_2011_z2la_for_qs101.copy()
        all_residents_scotland = all_residents_scotland[['Zone', '2011_LA', 'MSOA']]
        all_residents_scotland = pd.merge(all_residents_scotland,
                                          lad_scottish_base_year_all_pop,
                                          how='left',
                                          on='2011_LA')
        all_residents_scotland = pd.merge(all_residents_scotland,
                                          la_to_msoa_uk_lookup,
                                          how='left',
                                          on='MSOA')
        all_res_scot_cols_to_multiply = list(all_residents_scotland.columns)
        all_res_scot_cols_to_multiply = all_res_scot_cols_to_multiply[3:-1]
        all_residents_scotland[all_res_scot_cols_to_multiply] = all_residents_scotland[
            all_res_scot_cols_to_multiply].multiply(all_residents_scotland['lad_to_msoa'], axis='index')
        scot_pop_pcent_error = ((all_residents_scotland['Total'].sum() - scottish_pop_total) / scottish_pop_total) * 100
        self._logger.info('Summing the LAD to MSOA factors is: ' + str(all_residents_scotland['lad_to_msoa'].sum()))
        self._logger.info('It should be the sum of the number of Scottish districts (32).')
        if abs(scot_pop_pcent_error) < 0.0001:
            self._logger.info('After scaling the Scottish population to MSOA level,')
            self._logger.info('it has an error of ' + str(scot_pop_pcent_error) + '%.')
            self._logger.info('This is less than 0.0001% and is deemed an acceptable level of variation.')
        else:
            self._logger.warning('!!!!! WARNING !!!!!')
            self._logger.warning('After scaling the Scottish population data to MSOA level')
            self._logger.warning('an error of ' + str(scot_pop_pcent_error) + '% relative to the')
            self._logger.warning('input population was calculated. This is greater than')
            self._logger.warning('0.0001% and is deemed unacceptably large!')
            gen.print_w_toggle('!!!!! WARNING !!!!!', verbose=verbose)
            gen.print_w_toggle('After scaling the Scottish population data to MSOA level', verbose=verbose)
            gen.print_w_toggle('an error of', scot_pop_pcent_error, '% relative to the', verbose=verbose)
            gen.print_w_toggle('input population was calculated. This is greater than', verbose=verbose)
            gen.print_w_toggle('0.0001% and is deemed unacceptably large!', verbose=verbose)
        all_residents_scotland.drop(columns=['lad_to_msoa'], inplace=True)

        # Make MSOA (zonal) HHR table
        hhr_by_z = qs101_uk_by_z.copy()
        hhr_by_z = hhr_by_z[['%_of_HHR', 'MSOA']]

        # prepare England and Wales data for merging with Scotland and use in hhr
        ew_data_for_hhr = all_residents.copy()
        data_for_hhr_to_multiply = list(ew_data_for_hhr.columns)
        data_for_hhr_to_multiply = data_for_hhr_to_multiply[3:]
        ew_data_for_hhr[data_for_hhr_to_multiply] = ew_data_for_hhr[data_for_hhr_to_multiply].astype(float)

        # Need to add Scotland to all_residents before adding it to hhr
        scotland_data_for_hhr = all_residents_scotland.copy()
        uk_data_for_hhr = ew_data_for_hhr.append(scotland_data_for_hhr)

        # Merge the dta for hhr with the initial hhr setup and apply the % of HHR factor
        hhr_by_z = pd.merge(hhr_by_z, uk_data_for_hhr, how='left', on='MSOA')
        hhr_by_z[data_for_hhr_to_multiply] = hhr_by_z[data_for_hhr_to_multiply].multiply(hhr_by_z['%_of_HHR'],
                                                                                         axis='index')

        # Getting workers/SOC

        # Need to have LA level over 16s as a proportion of total pop
        # Later on can multiply this by APS proportion of workers who are over 16
        working_age_pop_by_la_uk = nomis_base_year_mye_pop_by_la_gb.copy()
        working_age_pop_by_la_uk = working_age_pop_by_la_uk[['2021_LA', 'Aged 16+', 'All Ages']]
        # 2018 data are (somehow!) formatted as strings with 1000 (comma) separators! 2019 is not.
        # Need to reformat to remove commas in 2018.
        working_age_pop_by_la_uk['Aged 16+'] = working_age_pop_by_la_uk['Aged 16+'].astype(str)
        working_age_pop_by_la_uk['Aged 16+'] = working_age_pop_by_la_uk['Aged 16+'].str.replace(',', '').astype(float)
        working_age_pop_by_la_uk['All Ages'] = working_age_pop_by_la_uk['All Ages'].astype(str)
        working_age_pop_by_la_uk['All Ages'] = working_age_pop_by_la_uk['All Ages'].str.replace(',', '').astype(float)
        # Drop blank lines if any have made it through the reformatting process - trying to stop Scilly taking over!
        working_age_pop_by_la_uk.dropna(axis=0, how='all', inplace=True)
        # Just create this column to hold the place for now
        working_age_pop_by_la_uk['Over_16_prop'] = 0

        aps_lad_lookup = aps_ftpt_gender_base_year_summary.copy()
        aps_lad_lookup = aps_lad_lookup[['LAD', '2021_LA']]
        working_age_pop_by_la_uk = pd.merge(aps_lad_lookup,
                                            working_age_pop_by_la_uk,
                                            how='right',
                                            on='2021_LA')

        working_age_pop_by_la_uk['LAD'].fillna("Isles of Scilly", inplace=True)

        aps_soc_to_merge = aps_soc_to_use.copy()
        aps_soc_to_merge = aps_soc_to_merge[['LAD', 'Aged_16+', 'Total_Workers']]
        working_age_pop_by_la_uk = pd.merge(working_age_pop_by_la_uk,
                                            aps_soc_to_merge,
                                            how='left',
                                            on='LAD')

        # Get totals to help solve Scilly
        working_age_pop_by_la_uk = working_age_pop_by_la_uk.append(
            working_age_pop_by_la_uk.sum(numeric_only=True), ignore_index=True)
        working_age_pop_by_la_uk['LAD'].fillna("UK wide total", inplace=True)
        working_age_pop_by_la_uk['2021_LA'].fillna("All_UK001", inplace=True)
        # Now the total column is added, populate this column
        working_age_pop_by_la_uk['Over_16_prop'] = (
                working_age_pop_by_la_uk['Aged 16+'] / working_age_pop_by_la_uk['All Ages'])

        working_age_pop_by_la_uk['APS_working_prop'] = (
                working_age_pop_by_la_uk['Total_Workers'] / working_age_pop_by_la_uk['Aged_16+'])
        working_age_pop_by_la_uk['worker/total_pop'] = (
                working_age_pop_by_la_uk['Over_16_prop'] * working_age_pop_by_la_uk['APS_working_prop'])

        working_age_pop_by_la_uk.set_index('LAD', inplace=True)
        working_age_pop_mask_nans = working_age_pop_by_la_uk.loc['Isles of Scilly', :].isnull()
        working_age_pop_by_la_uk.loc[
            'Isles of Scilly', working_age_pop_mask_nans] = working_age_pop_by_la_uk.loc[
            'UK wide total', working_age_pop_mask_nans]
        working_age_pop_by_la_uk.reset_index(inplace=True)

        pop_props_by_2021_la = working_age_pop_by_la_uk.copy()
        pop_props_by_2021_la = pop_props_by_2021_la[['LAD', '2021_LA', 'worker/total_pop']]
        pop_props_by_2021_la = pd.merge(pop_props_by_2021_la,
                                        aps_props_to_merge,
                                        how='outer',
                                        on='2021_LA')

        # Summarise HHR data by district
        hhr_by_d = hhr_by_z.copy()
        hhr_by_d = hhr_by_d.groupby(['2011_LA']).sum().reset_index()
        hhr_by_d.drop(columns=['%_of_HHR', 'Zone'], inplace=True)

        # Switch to 2021_LA
        la2011_to_la2021 = uk_2011_and_2021_la.copy()
        la2011_to_la2021 = la2011_to_la2021[['2011 LA Code', '2021 LA Code']]
        la2011_to_la2021.columns = la2011_to_la2021.columns.str.replace(' LA Code', '_LA', regex=True)
        hhr_by_d = pd.merge(la2011_to_la2021, hhr_by_d, how='right', on='2011_LA')
        hhr_by_d.drop(columns=['2011_LA'], inplace=True)
        hhr_by_d = hhr_by_d.groupby(['2021_LA']).sum().reset_index()

        # Produce worker table
        hhr_worker_by_d = hhr_by_d.copy()
        hhr_worker_by_d = hhr_worker_by_d[['2021_LA', 'Total']]
        hhr_worker_by_d['Worker'] = 0  # Create column in correct place. We'll fill it later
        hhr_worker_by_d = pd.merge(hhr_worker_by_d, pop_props_by_2021_la, how='left', on='2021_LA')
        hhr_worker_by_d.drop(columns=['LAD'], inplace=True)
        hhr_worker_by_d['Worker'] = hhr_worker_by_d['Total'] * hhr_worker_by_d['worker/total_pop']
        hhr_worker_by_d.drop(columns=['Total', 'worker/total_pop'], inplace=True)
        hhr_worker_type = hhr_worker_by_d.columns
        hhr_worker_type = hhr_worker_type[2:]
        hhr_worker_by_d[hhr_worker_type] = hhr_worker_by_d[hhr_worker_type].multiply(hhr_worker_by_d['Worker'],
                                                                                     axis='index')
        hhr_worker_by_d['Checksum'] = ((hhr_worker_by_d['Worker'] * 2) -
                                       hhr_worker_by_d[hhr_worker_type].sum(axis=1))

        if (abs(max(hhr_worker_by_d['Checksum'])) < 0.000001 and
                abs(min(hhr_worker_by_d['Checksum'])) < 0.000001):
            self._logger.info('Worker proportions summed to total')
            self._logger.info('across both ft/pt by gender and SOC')
            self._logger.info('(within reasonable deviation)')
            self._logger.info('Max deviation value was: %s ' % str(max(hhr_worker_by_d['Checksum'])))
            self._logger.info('Min deviation value was: %s ' % str(min(hhr_worker_by_d['Checksum'])))
        else:
            self._logger.warning('!!!!! WARNING !!!!!')
            self._logger.warning('Worker proportions did not sum to total')
            self._logger.warning('across both ft/pt by gender and SOC')
            self._logger.warning('(within reasonable deviation)')
            self._logger.warning('Max deviation value was: %s ' % str(max(hhr_worker_by_d['Checksum'])))
            self._logger.warning('Min deviation value was: %s ' % str(min(hhr_worker_by_d['Checksum'])))
            gen.print_w_toggle('!!!!! WARNING !!!!!', verbose=verbose)
            gen.print_w_toggle('Worker proportions did not sum to total', verbose=verbose)
            gen.print_w_toggle('across both ft/pt by gender and SOC', verbose=verbose)
            gen.print_w_toggle('(within reasonable deviation)', verbose=verbose)
            gen.print_w_toggle('Max deviation value was:', max(hhr_worker_by_d['Checksum']), verbose=verbose)
            gen.print_w_toggle('Min deviation value was:', min(hhr_worker_by_d['Checksum']), verbose=verbose)
        hhr_worker_by_d_for_export = hhr_worker_by_d.copy()
        hhr_worker_by_d_for_export = hhr_worker_by_d_for_export[['2021_LA',
                                                                 'Worker',
                                                                 'male fte',
                                                                 'male pte',
                                                                 'female fte',
                                                                 'female pte',
                                                                 'higher',
                                                                 'medium',
                                                                 'skilled']]
        hhr_worker_by_d_for_export.columns = hhr_worker_by_d_for_export.columns.str.replace('male', 'Male', regex=True)
        hhr_worker_by_d_for_export.columns = hhr_worker_by_d_for_export.columns.str.replace('feMale', 'Female',
                                                                                            regex=True)
        hhr_worker_by_d_for_export.columns = hhr_worker_by_d_for_export.columns.str.replace('_LA', '_LA_code',
                                                                                            regex=True)

        hhr_worker_by_d_row_info = nomis_base_year_mye_pop_by_la_gb.copy()
        hhr_worker_by_d_row_info.rename(
            columns={'local authority: district / unitary (as of April 2021)': '2021_LA_name',
                     '2021_LA': '2021_LA_code'},
            inplace=True)
        hhr_worker_by_d_row_info = hhr_worker_by_d_row_info[['2021_LA_name', '2021_LA_code']]
        hhr_worker_by_d_row_info['LA'] = hhr_worker_by_d_row_info.index + 1

        hhr_worker_by_d_for_export = pd.merge(hhr_worker_by_d_row_info,
                                              hhr_worker_by_d_for_export,
                                              how='left',
                                              on='2021_LA_code')

        # Produce Non-worker table
        hhr_nonworker_by_d = hhr_by_d.copy()
        hhr_nonworker_by_d = pd.merge(hhr_nonworker_by_d, hhr_worker_by_d, how='left', on='2021_LA')
        hhr_nonworker_by_d_cols_to_rem = hhr_nonworker_by_d.columns
        hhr_nonworker_by_d_cols_to_rem = hhr_nonworker_by_d_cols_to_rem[1:]
        hhr_nonworker_by_d['Non worker'] = hhr_nonworker_by_d['Total'] - hhr_nonworker_by_d['Worker']
        hhr_nonworker_by_d['Children'] = hhr_nonworker_by_d['M_under_16'] + hhr_nonworker_by_d['F_under_16']
        hhr_nonworker_by_d['M_75 and over'] = hhr_nonworker_by_d['M_75_and_over']
        hhr_nonworker_by_d['F_75 and over'] = hhr_nonworker_by_d['F_75_and_over']
        hhr_nonworker_by_d['M_16-74_out'] = (hhr_nonworker_by_d['M_16-74'] -
                                             (hhr_nonworker_by_d['male fte'] +
                                              hhr_nonworker_by_d['male pte']))
        hhr_nonworker_by_d['F_16-74_out'] = (hhr_nonworker_by_d['F_16-74'] -
                                             (hhr_nonworker_by_d['female fte'] +
                                              hhr_nonworker_by_d['female pte']))
        pe_dag = hhr_nonworker_by_d.copy()  # Copy the nonworker df here - it has all pop cols needed in Pe_dag audit later
        hhr_nonworker_by_d = hhr_nonworker_by_d.drop(columns=hhr_nonworker_by_d_cols_to_rem)
        hhr_nonworker_by_d.rename(columns={'2021_LA': '2021_LA_code'}, inplace=True)
        hhr_nonworker_by_d.columns = hhr_nonworker_by_d.columns.str.rstrip('_out')
        hhr_nonworker_types_by_d = hhr_nonworker_by_d.columns
        hhr_nonworker_types_by_d = hhr_nonworker_types_by_d[2:]
        hhr_nonworker_by_d['Checksum'] = (hhr_nonworker_by_d['Non worker'] -
                                          hhr_nonworker_by_d[hhr_nonworker_types_by_d].sum(axis=1))
        if (abs(max(hhr_nonworker_by_d['Checksum'])) < 0.000001 and
                abs(min(hhr_nonworker_by_d['Checksum'])) < 0.000001):
            self._logger.info('Non-worker proportions summed to total across both all non-worker types')
            self._logger.info('(within reasonable deviation)')
            self._logger.info('Max deviation value was: %s ' % str(max(hhr_nonworker_by_d['Checksum'])))
            self._logger.info('Min deviation value was: %s ' % str(min(hhr_nonworker_by_d['Checksum'])))
        else:
            self._logger.warning('!!!!! WARNING !!!!!')
            self._logger.warning('Non-worker proportions did not sum to total')
            self._logger.warning('across both all non-worker types')
            self._logger.warning('(within reasonable deviation)')
            self._logger.warning('Max deviation value was: %s ' % str(max(hhr_nonworker_by_d['Checksum'])))
            self._logger.warning('Min deviation value was: %s ' % str(min(hhr_nonworker_by_d['Checksum'])))
            gen.print_w_toggle('!!!!! WARNING !!!!!', verbose=verbose)
            gen.print_w_toggle('Non-worker proportions did not sum to total', verbose=verbose)
            gen.print_w_toggle('across both all non-worker types', verbose=verbose)
            gen.print_w_toggle('(within reasonable deviation)', verbose=verbose)
            gen.print_w_toggle('Max deviation value was:', max(hhr_nonworker_by_d['Checksum']), verbose=verbose)
            gen.print_w_toggle('Min deviation value was:', min(hhr_nonworker_by_d['Checksum']), verbose=verbose)
        hhr_nonworker_by_d.drop(columns=['Checksum'], inplace=True)

        hhr_nonworker_by_d_for_export = pd.merge(hhr_worker_by_d_row_info,
                                                 hhr_nonworker_by_d,
                                                 how='left',
                                                 on='2021_LA_code')

        # Produce a Pe_(d, a, g) df for use in auditing Step 3.2.9 later
        pe_dag = pe_dag.rename(columns={'2021_LA': '2021_LA_code'})
        pe_dag = pd.melt(pe_dag, id_vars=['2021_LA_code'], value_vars=[
            'Children', 'M_16-74', 'F_16-74', 'M_75 and over', 'F_75 and over']).rename(
            columns={'variable': 'ag', 'value': 'HHR_pop'})
        pe_dag['a'] = pe_dag['ag'].map(const.A)
        pe_dag['g'] = pe_dag['ag'].map(const.G)
        pe_dag = pe_dag[['2021_LA_code', 'a', 'g', 'HHR_pop']]
        # TODO(NK) : Check this place. We currently dont have anything called mye_aps_process_dir
        """
        full_mye_aps_process_dir = os.path.join(self.out_paths['write_folder'],
                                                process_dir,
                                                mye_aps_process_dir)
        hhr_vs_all_pop_name = '_'.join(['gb', self.model_zoning.lower(),
                                        ModelYear, 'pop+hh_pop.csv'])
        hhr_worker_by_d_for_export_name = '_'.join(['mye_gb_d', ModelYear, 'wkrs_tot+by_ag.csv'])
        hhr_nonworker_by_d_for_export_name = '_'.join(['mye_gb_d', ModelYear, 'nwkrs_tot+by_ag.csv'])
        la_info_for_2021_name = r'lookup_gb_2021_lad_to_d.csv'
        """

        gen.safe_dataframe_to_csv(pe_dag, self.hhr_pop_by_dag)

        # Export only the requested outputs
        mye_aps_logging_string = 'The MYE_APS_process completed after being called by'
        if called_by == 'MYE_pop_compiled':
            hhr_vs_all_pop = hhr_by_z.copy()
            hhr_vs_all_pop = hhr_vs_all_pop[['Zone', 'MSOA', 'Total']]
            hhr_vs_all_pop.rename(columns={'Total': 'Total_HHR'}, inplace=True)
            hhr_vs_all_pop = pd.merge(hhr_vs_all_pop, uk_data_for_hhr[['MSOA', 'Total']], how='left', on='MSOA')
            hhr_vs_all_pop.rename(columns={'Total': 'Total_Pop'}, inplace=True)
            # Now write out MYE_MSOA_pop as step 3.2.10 (and 3.2.11?) need it.
            # As it is only 8480 lines long, it should be quick to write/read
            # It saves both of these steps from having to call step 3.2.5 to recalculate it.
            # Unlike this step (step 3.2.5), steps 3.2.10/3.2.11 will be called
            # after this function (and 3.2.5) have run, so will not need to skip ahead
            # to call it, as was the case here.

            gen.safe_dataframe_to_csv(hhr_vs_all_pop, self.hhr_vs_all_pop_path, index=False)
            mye_aps_process_output = hhr_vs_all_pop
        elif called_by == 'LA_level_adjustment':
            # Dump the APS worker and non-worker files
            gen.safe_dataframe_to_csv(hhr_worker_by_d_for_export, self.hhr_worker_by_d_for_export_path,
                                      index=False)
            gen.safe_dataframe_to_csv(hhr_nonworker_by_d_for_export, self.hhr_nonworker_by_d_for_export_path,
                                      index=False)
            la_info_for_2021 = hhr_nonworker_by_d_for_export.copy()
            la_info_for_2021 = la_info_for_2021[['2021_LA_name', '2021_LA_code', 'LA']]
            gen.safe_dataframe_to_csv(la_info_for_2021, self.la_info_for_2021_path, index=False)
            # Create a list of outputs that can be picked up by the calling function
            mye_aps_process_output = [hhr_worker_by_d_for_export,
                                      hhr_nonworker_by_d_for_export,
                                      la_info_for_2021,
                                      pe_dag]
        else:
            self._logger.info('WARNING - The function that called the MYE_APS_process is not recognised!')
            self._logger.info('Why was the MYE_APS_process called by an unrecognised function?')
            # Dump all residents table for Scotland - Not used but kept for QA purposes
            # all_residents_scotland.to_csv(
            #     os.path.join(full_mye_aps_process_dir, all_residents_scotland_name), index = False)
            mye_aps_process_output = ['No data to export', 'See warning for MYE_APS_Function']
        self._logger.info('%s %s' % (mye_aps_logging_string, called_by))
        self._logger.info('Note that only the outputs requested by the function that called it have been generated')
        gen.print_w_toggle('The MYE_APS_process function completed', verbose=verbose)
        gen.print_w_toggle('Note that only the outputs requested by the function that called it have been generated',
              verbose=verbose)

        return mye_aps_process_output

    def ntem_pop_interpolation(self,
                               verbose: bool = True
                               ) -> pd.DataFrame:

        """
        Process population data from NTEM CTripEnd database:
        Interpolate population to the target year, in this case it is for the base year, as databases
        are available in 5 year interval;
        Translate NTEM zones in Scotland into NorNITs zones; for England and Wales,
        NTEM zones = NorMITs zones (MSOAs)

        Parameters
        ----------
        verbose: bool = True
            Whether to print text or not.

        Return
        ------
        ntem_hhpop: pd.Dataframe
        """

        # The year of data is set to define the upper and lower NTEM run years and interpolate as necessary between them.
        # The base year for NTEM is 2011 and it is run in 5-year increments from 2011 to 2051.
        # The year selected below must be between 2011 and 2051 (inclusive).
        # As we are running this inside the main base year script, we can set Year = ModelYear
        # However, we do still need to retain Year, as it is assumed Year is an int, not a str (as ModelYear is).
        year = int(self.base_year)

        self._logger.info('Running NTEM_Pop_Interpolation function for Year: %d ' % year)
        gen.print_w_toggle('Running NTEM_Pop_Interpolation function for Year: %d ' % year, verbose=verbose)

        if year < 2011 | year > 2051:
            raise ValueError("Please enter a valid year of data.")
        else:
            pass

        self._logger.info('NTEM_Pop_Interpolation output being written in:')

        self._logger.info(self.write_folder)
        out_lines = [
            'Notebook run on - %s' % str(timing.get_datetime()),
            'Data Year - %s' % self.base_year,
            'Correspondence Lists:',
            self.zone_path,
            self.pop_seg_path,
        ]

        with open(self.ntem_logfile_path, 'w') as text_file:
            text_file.write('\n'.join(out_lines))
        # 'I:/NorMITs Synthesiser/Zone Translation/'

        # Data years
        # NTEM is run in 5-year increments with a base of 2011.
        # This section calculates the upper and lower years of data that are required
        interpolationyears = year % 5
        loweryear = year - ((interpolationyears - 1) % 5)
        upperyear = year + ((1 - interpolationyears) % 5)

        self._logger.info("Lower Interpolation Year - %s" % str(loweryear))
        self._logger.info("Upper Interpolation Year - %s" % str(upperyear))
        gen.print_w_toggle("Lower Interpolation Year - %s" % str(loweryear), verbose=verbose)
        gen.print_w_toggle("Upper Interpolation Year - %s" % str(upperyear), verbose=verbose)

        # Import Upper and Lower Year Tables
        # 'I:/Data/NTEM/NTEM 7.2 outputs for TfN/'

        LowerNTEMDatabase = os.path.join(const.CTripEnd_Database, const.CTripEnd % loweryear)
        UpperNTEMDatabase = os.path.join(const.CTripEnd_Database, const.CTripEnd % upperyear)
        # UpperNTEMDatabase = self.CTripEnd_Database_path + r"\CTripEnd7_" + str(upperyear) + r".accdb"

        cnxn = pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' +
                              '{};'.format(UpperNTEMDatabase))

        query = r"SELECT * FROM ZoneData"
        uzonedata = pd.read_sql(query, cnxn)
        cnxn.close()

        cnxn = pyodbc.connect('DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' +
                              '{};'.format(LowerNTEMDatabase))

        query = r"SELECT * FROM ZoneData"
        lzonedata = pd.read_sql(query, cnxn)
        cnxn.close()

        # Re-format Tables
        lzonepop = lzonedata.copy()
        uzonepop = uzonedata.copy()
        lzonepop.drop(const.POP_DROP, axis=1, inplace=True)
        uzonepop.drop(const.POP_DROP, axis=1, inplace=True)
        lzonepop_long = pd.melt(lzonepop, id_vars=const.POP_COL,
                                var_name="LTravellerType", value_name="LPopulation")
        uzonepop_long = pd.melt(uzonepop, id_vars=const.POP_COL,
                                var_name="UTravellerType", value_name="UPopulation")

        lzonepop_long.rename(columns={"I": "LZoneID", "B": "LBorough", "R": "LAreaType"}, inplace=True)
        uzonepop_long.rename(columns={"I": "UZoneID", "B": "UBorough", "R": "UAreaType"}, inplace=True)

        lzonepop_long['LIndivID'] = lzonepop_long.LZoneID.map(str) + "_" + lzonepop_long.LAreaType.map(
            str) + "_" + lzonepop_long.LBorough.map(str) + "_" + lzonepop_long.LTravellerType.map(str)
        uzonepop_long['UIndivID'] = uzonepop_long.UZoneID.map(str) + "_" + uzonepop_long.UAreaType.map(
            str) + "_" + uzonepop_long.UBorough.map(str) + "_" + uzonepop_long.UTravellerType.map(str)

        # Join Upper and Lower Tables
        tzonepop_datayear = lzonepop_long.join(uzonepop_long.set_index('UIndivID'), on='LIndivID', how='right',
                                               lsuffix='_left', rsuffix='_right')
        tzonepop_datayear.drop(['UZoneID', 'UBorough', 'UAreaType', 'UTravellerType'], axis=1, inplace=True)

        # Interpolate Between Upper and Lower Years
        tzonepop_datayear['GrowthinPeriod'] = tzonepop_datayear.eval('UPopulation - LPopulation')
        tzonepop_datayear['GrowthperYear'] = tzonepop_datayear.eval('GrowthinPeriod / 5')
        tzonepop_datayear = tzonepop_datayear.assign(
            GrowthtoYear=tzonepop_datayear['GrowthperYear'] * (year - loweryear))
        tzonepop_datayear['Population'] = tzonepop_datayear.eval('LPopulation + GrowthtoYear')

        # Tidy up
        tzonepop_datayear.rename(columns=const.TZONEPOP_COL_RENAME, inplace=True)
        tzonepop_datayear.drop(const.TZONEPOP_COL_DROP, axis=1, inplace=True)
        gen.print_w_toggle(tzonepop_datayear.Population.sum(), verbose=verbose)

        # Translating zones for those in Scotland
        zone_list = gen.safe_read_csv(self.zone_path)
        tzonepop_datayear = tzonepop_datayear.join(zone_list.set_index('ntemZoneID'), on='ZoneID', how='right')
        # tzonepop_datayear.rename(columns={'msoaZoneID': 'ModelZone'}, inplace=True)
        tzonepop_datayear[
            'Population_RePropped'] = tzonepop_datayear['Population'] * tzonepop_datayear[
            'overlap_ntem_pop_split_factor']

        segmentation_list = gen.safe_read_csv(self.pop_seg_path)
        tzonepop_datayear = tzonepop_datayear.join(segmentation_list.set_index('NTEM_Traveller_Type'),
                                                   on='TravellerType',
                                                   how='right')
        tzonepop_datayear.drop(const.TZONEPOP_COLS_DROP, axis=1, inplace=True)
        tzonepop_datayear.rename(columns={"Population_RePropped": "Population"}, inplace=True)
        gen.print_w_toggle(tzonepop_datayear.Population.sum(), verbose=verbose)
        tzonepop_datayear = tzonepop_datayear.groupby(const.TZONEPOP_GROUPBY)[['Population']].sum().reset_index()
        ntem_hhpop = tzonepop_datayear
        # Export
        export_summarypop = tzonepop_datayear.groupby(['TravellerType', 'NTEM_TT_Name']).sum()
        gen.print_w_toggle(export_summarypop.Population.sum(), verbose=verbose)
        # Export_SummaryPop.drop(['msoaZoneID'], inplace=True, axis=1)

        if self.export_process_files:
            gen.safe_dataframe_to_csv(tzonepop_datayear, self.popoutput_path, index=False, sep=",")

        with open(self.ntem_logfile_path, "a") as o:
            o.write("Total Population: \n")
            export_summarypop.to_csv(o, header=False, sep="-")
            o.write("\n")

        gen.print_w_toggle("Export complete.", verbose=verbose)
        gen.print_w_toggle(ntem_hhpop.head(5), verbose=verbose)
        self._logger.info('NTEM_Pop_Interpolation function complete')
        gen.print_w_toggle('NTEM_Pop_Interpolation function complete', verbose=verbose)
        return ntem_hhpop

    def mye_pop_compiled(self,
                         crp: pd.DataFrame,
                         read_by_pop_msoa_file: bool = False,
                         verbose: bool = True,
                         ) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Adjusts base year population according to Mid Year Population Estimate.

        Splits total base year (2018 or 2019) population from MYPE sourced from
        NOMIS into household residents and residents living in communal
        establishment using ratio derived from Census 2011. Then adjusts NTEM
        2018/2019 household population (extracted from CTripEnd Access database)
        and the estimated 2018/2019 household population from previous step to be
        consistent with ONS MYE population estimates on base year (2018 or 2019)
        on household residents.

        Parameters
        ----------
        crp: pd.Dataframe
            Dataframe containing estimated household population.

        read_by_pop_msoa_file: bool = False
            Option to read in data processed by step the APS compiling function
            through an existing csv.

        verbose: bool = True
            Whether to print text or not.

        Return
        ------
        aj_crp: pd.Dataframe
            Dataframe containing MYPE compliant household population segmented
            by zone and dwelling type.

        ntem_hh_pop: pd.Dataframe
            Dataframe containing MYPE compliant NTEM household population
            segmented by zone, age, gender, household composition and
            employment status.

        audit_mye_msoa_pop: pd.Dataframe
            Dataframe containing zonal household population returned from
            mye_aps_process() for audit purpose.

     """

        self._logger.info('Running Step 3.2.5')
        gen.print_w_toggle('Running Step 3.2.5', verbose=verbose)

        # Adam - DONE: the following inputs from MYE should be called in from your outcome of function MYE_APS_process
        # Note that this should work regardless of the position of the function that is being called being later in the doc
        # string as the function gets called directly from this script.
        # It's a bit inefficient as every function that requires MYE_APS_process outputs runs it!
        # TODO - Control in MYE_APS_process which tells it which function has called it and thus which part of
        #  the function to run and export and then alters the various calls and readings of outputs to work with this
        #  altered format. Some work has been done on this to prevent wasting time exporting the outputs several times,
        #  but it could be made much more efficient!
        # Added a manual control to allow read in from file (rather than internal memory) in the event of a partial run
        # False means "read in from memory" and is the default.

        if read_by_pop_msoa_file:
            mye_msoa_pop = gen.safe_read_csv(self.mye_msoa_pop)

            self._logger.info(
                'Step 3.2.5 read in data processed by step the APS compiling function through an existing csv')
            self._logger.info('WARNING - This is not the default way of reading this data!')
            self._logger.info('Did you mean to do that?')
        else:
            self._logger.info(
                'Step 3.2.5 is calling step the APS compiling function in order to obtain Base Year population data')

            mye_msoa_pop = self.mye_aps_process(called_by=const.MYE_POP_COMPILED_NAME, )
            self._logger.info('Step 3.2.5 successfully read in data processed by the MYE_APS_process function')

            self._logger.info('from internal memory')

        audit_mye_msoa_pop = mye_msoa_pop.copy()

        # Call Step 3.2.4 to get crp_pop
        self._logger.info('Step 3.2.5 is calling Step 3.2.4 to get crp_pop')
        gen.print_w_toggle('Step 3.2.5 is calling Step 3.2.4 to get crp_pop', verbose=verbose)
        crp_pop = self.property_type_refactor(crp)
        self._logger.info('Step 3.2.5 has called Step 3.2.4 and has obtained crp_pop')
        gen.print_w_toggle('Step 3.2.5 has called Step 3.2.4 and has obtained crp_pop', verbose=verbose)

        crp_msoa_pop = crp_pop.groupby(['ZoneID'])['population'].sum().reset_index()
        crp_msoa_pop = crp_msoa_pop.merge(mye_msoa_pop, how='left', left_on='ZoneID', right_on='MSOA').drop(
            columns={'MSOA'})
        # print(crp_msoa_pop.head(5))
        crp_msoa_pop['pop_aj_factor'] = crp_msoa_pop['Total_HHR'] / crp_msoa_pop['population']
        crp_msoa_pop = crp_msoa_pop.drop(columns={'Total_HHR', 'Total_Pop', 'population'})
        # print(crp_msoa_pop.head(5))
        aj_crp = crp_pop.merge(crp_msoa_pop, how='left', on='ZoneID')
        aj_crp['aj_population'] = aj_crp['population'] * aj_crp['pop_aj_factor']
        aj_crp = aj_crp.drop(columns={'population'})
        aj_crp = aj_crp.rename(columns={'aj_population': 'population'})
        audit_aj_crp = aj_crp.copy()
        # print(aj_crp.head(5))

        # Start of block moved from Step 3.2.6 due to need to audit in Step 3.2.5

        # Car availability from NTEM
        # Read NTEM hh pop at NorMITs Zone level and make sure the zonal total is consistent to crp
        # TODO(NK): Check this as we havent use mye_pop_compiled_dir so far
        """
        ntem_hh_pop = ntem_pop_interpolation(self, mye_pop_compiled_dir)

        uk_msoa = gpd.read_file(_default_msoaRef)[['objectid', 'msoa11cd']]
        """
        ntem_hh_pop = self.ntem_pop_interpolation()
        uk_msoa = gen.safe_read_csv(const.DEFAULT_MSOAREF, usecols=['objectid', 'msoa11cd'])

        ntem_hh_pop = ntem_hh_pop.merge(uk_msoa, how='left', left_on='msoaZoneID', right_on='objectid')

        ntem_hhpop_cols_to_groupby = const.NTEM_HH_POP_COLS[:-1]
        ntem_hh_pop = ntem_hh_pop.groupby(ntem_hhpop_cols_to_groupby)['Population'].sum().reset_index()

        # Testing with Manchester
        # NTEM_HHpop_E02001045 = ntem_hh_pop[ntem_hh_pop['msoa11cd'] == 'E02001045']
        # NTEM_HHpop_E02001045.to_csv('NTEM_HHpop_E02001045.csv', index=False)

        ntem_hh_pop = ntem_hh_pop[const.NTEM_HH_POP_COLS]
        ntem_hhpop_total = ntem_hh_pop.groupby(['msoaZoneID'])['Population'].sum().reset_index()
        ntem_hhpop_total = ntem_hhpop_total.rename(columns={'Population': 'ZoneNTEMPop'})
        # print('Headings of ntem_hhpop_total')
        # print(ntem_hhpop_total.head(5))
        if self.export_process_files:
            gen.safe_dataframe_to_csv(ntem_hhpop_total, self.ntem_hhpop_total_path, index=False)

        hhpop_dt_total = aj_crp.groupby(['ZoneID'])['population'].sum().reset_index()
        hhpop_dt_total = hhpop_dt_total.rename(columns={'population': 'ZonePop'})
        # print('Headings of hhpop_dt_total')
        # print(hhpop_dt_total.head(5))
        if self.export_process_files:
            gen.safe_dataframe_to_csv(hhpop_dt_total, self.hhpop_dt_total_path, index=False)

        ntem_hh_pop = ntem_hh_pop.merge(ntem_hhpop_total, how='left', on=['msoaZoneID'])
        ntem_hh_pop = ntem_hh_pop.merge(hhpop_dt_total, how='left', left_on=['msoa11cd'],

                                        right_on=['ZoneID']).drop(columns={'ZoneID'})
        # print('Headings of ntem_hh_pop')
        # print(ntem_hh_pop.head(5))
        ntem_hh_pop['pop_aj_factor'] = ntem_hh_pop['ZonePop'] / ntem_hh_pop['ZoneNTEMPop']

        ntem_hh_pop['pop_aj'] = ntem_hh_pop['Population'] * ntem_hh_pop['pop_aj_factor']
        audit_ntem_hhpop = ntem_hh_pop.copy()
        # print(ntem_hh_pop.pop_aj.sum())
        # print(aj_crp.population.sum())

        # End of block moved from Step 3.2.6 due to need to audit in Step 3.2.5

        self._logger.info('Total population from MYE: ')
        self._logger.info(mye_msoa_pop.Total_Pop.sum())
        self._logger.info('Total household residents from MYE: ')
        self._logger.info(mye_msoa_pop.Total_HHR.sum())
        self._logger.info('Total household residents from aj_crp: ')
        self._logger.info(aj_crp.population.sum())
        self._logger.info('Population currently {}'.format(aj_crp.population.sum()))

        if self.export_process_files:
            gen.safe_dataframe_to_csv(aj_crp, self.mye_pop_compiled_path, index=False)

        # Audit
        audits.audit_3_2_5(self, audit_aj_crp, audit_ntem_hhpop, audit_mye_msoa_pop, mye_msoa_pop)

        if self.export_process_files:
            compress.write_out(ntem_hh_pop, self.ntem_hh_pop_path)

        self._logger.info('Step 3.2.5 completed')
        gen.print_w_toggle('Step 3.2.5 completed', verbose=verbose)
        return [aj_crp, ntem_hh_pop, audit_mye_msoa_pop]

    def pop_with_full_dimensions(self,
                                 compiled_mye_pop: list[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                                 verbose: bool = True
                                 ) -> None:

        """
        Function to join the bespoke census query to the classified residential property data.

        Problem here is that there are segments with attributed population coming in from the bespoke census query that
        don't have properties to join on. So we need classified properties by MSOA for this to work atm
        TODO: make it work for zones other than MSOA?
        ART, 12/11/2021 model_year moved defined at the head of this file. Can't find model_name!

        Parameters
        ----------
        compiled_mye_pop: list[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Output from mye_pop_compiled().

        verbose: bool = True
            Whether to print text or not.

        Return
        ------
        None.
        """
        self._logger.info('Running Step 3.2.6/3.2.7 function')
        gen.print_w_toggle('Running Step 3.2.6/3.2.7 function', verbose=verbose)

        # TODO - Sort out these filepaths a bit more permanently - may be redundant long term if NorCOM is integrated

        # TODO - Make this switching more robust
        # Control now in by_lu.py
        how_to_run = self.norcom
        gen.print_w_toggle(how_to_run, verbose=verbose)

        self._logger.info('Running with section 3.2.6 set to %s' % how_to_run)

        # TODO - These variable names could probably do with tidying up at bit to avoid confusion with
        #  their Step 3_2_5 counterparts. Only still like this as a block of code was moved from Step
        #  3_2_6 to Step 3_2_5 at the last minute!
        self._logger.info('Step 3.2.6 is calling Step 3.2.5')
        gen.print_w_toggle('Step 3.2.6 is calling Step 3.2.5', verbose=verbose)
        aj_crp, ntem_hhpop, audit_original_hhpop = compiled_mye_pop

        self._logger.info('Step 3.2.6 has completed its call of Step 3.2.5, Step 3.2.6 is continuing...')
        gen.print_w_toggle('Step 3.2.6 has completed its call of Step 3.2.5, Step 3.2.6 is continuing...',
                           verbose=verbose)

        if how_to_run == 'export to NorCOM':
            gen.print_w_toggle('Dumping NorCOM output file to NorCOM input directory and main LU output directory...',
                               verbose=verbose)
            gen.safe_dataframe_to_csv(ntem_hhpop, self.output_ntem_hhpop_filepath, index=False)
            gen.safe_dataframe_to_csv(ntem_hhpop, self.output_ntem_hhpop_out_path, index=False)
            gen.print_w_toggle('Output dumped - script should be ending now...', verbose=verbose)

            # Testing for just Manchester
            # NTEM_HHpop_Aj_E02001045 = NTEM_HHpop[NTEM_HHpop['msoa11cd'] == 'E02001045']
            # NTEM_HHpop_Aj_E02001045.to_csv('_'.join([
            #     'output_0_ntem_gb_msoa_ntem_tt', ModelYear, 'aj_hh_pop_E02001045.csv']), index=False)
            self._logger.info('Step 3.2.6 completed')
            self._logger.info('Dumped file %s for NorCOM' % const.OUTPUT_NTEM_HHPOP_FNAME)
            msg = (
                '!!!!!! SERIOUS WARNING !!!!!!\n'
                'Any further functions that are called from this Land Use process are highly likely to be wrong!'
            )
            self._logger.warning(msg)
            warnings.warn(msg)
            # TODO - Call NorCOM script directly here? Then could remove if/else statement?
            #  Waiting for NorCOM to be refined enough that it doesn't take 3 days to run...
        else:
            self._logger.info('Reading in file from NorCOM')
            # Make sure this read in and is picking up an actual file and actual columns
            norcom_ntem_hhpop = gen.safe_read_csv(self.input_ntem_hhpop_path,
                                                  usecols=const.NorCOM_NTEM_HHpop_col)

            # Sort out df prior to merger
            ntem_hhpop_trim = ntem_hhpop[const.NTEM_HHPOP_COLS]
            ntem_hhpop_trim.groupby(const.NTEM_HHPOP_COLS)

            ntem_hhpop_trim = pd.merge(ntem_hhpop_trim,
                                       norcom_ntem_hhpop,
                                       how='right',
                                       left_on=['msoa11cd', 'TravellerType'],
                                       right_on=['msoa11cd', 'lu_TravellerType'])

            ntem_hhpop_trim = ntem_hhpop_trim.rename(columns=const.NTEM_HHPOP_COL_RENAME)
            ntem_hhpop_trim['z'] = ntem_hhpop_trim['z'].astype(int)
            ntem_hhpop_trim_iterator = zip(ntem_hhpop_trim['z'],
                                           ntem_hhpop_trim['a'],
                                           ntem_hhpop_trim['g'],
                                           ntem_hhpop_trim['h'],
                                           ntem_hhpop_trim['e'])

            ntem_hhpop_trim['aghe_Key'] = ['_'.join([str(z), str(a), str(g), str(h), str(e)])
                                           for z, a, g, h, e in ntem_hhpop_trim_iterator]

            # Read in f (f_tns|zaghe) to expand adjustedNTEM hh pop with additional dimension of t(dwelling type),
            # n(HRP NS-SEC) and s (SOC)
            # Replace this block with new process from 2011 output f.
            # ['z', 'a', 'g', 'h', 'e', 't', 'n', 's', 'f_tns|zaghe']

            # This is the 2011 Census Data coming in

            census_f_value = gen.safe_read_csv(self.census_f_value_path)

            # census_f_value['z'] = census_f_value['z'].astype(int)
            census_f_value_iterator = zip(census_f_value['z'],
                                          census_f_value['a'],
                                          census_f_value['g'],
                                          census_f_value['h'],
                                          census_f_value['e'])
            census_f_value['aghe_Key'] = ['_'.join([str(z), str(a), str(g), str(h), str(e)])
                                          for z, a, g, h, e in census_f_value_iterator]
            ntem_hhpop_trim = pd.merge(ntem_hhpop_trim,
                                       census_f_value,
                                       on='aghe_Key')
            ntem_hhpop_trim = ntem_hhpop_trim.drop(
                columns=['z_x', 'a_x', 'g_x', 'h_x', 'e_x', 'lu_TravellerType'])
            ntem_hhpop_trim = ntem_hhpop_trim.rename(
                columns={'z_y': 'z', 'a_y': 'a', 'g_y': 'g', 'h_y': 'h', 'e_y': 'e'})
            ntem_hhpop_trim['P_aghetns'] = (ntem_hhpop_trim['f_tns|zaghe']
                                            * ntem_hhpop_trim['P_NTEM'])

            # Testing with Manchester
            # NTEM_HH_P_aghetns_E02001045 = ntem_hhpop_trim[ntem_hhpop_trim['msoa11cd'] == 'E02001045']
            # NTEM_HH_P_aghetns_E02001045.to_csv('NTEM_HH_P_aghetns_E02001045.csv', index=False)
            self._logger.info('NTEM HH pop scaled by f is currently:')
            self._logger.info(ntem_hhpop_trim.P_aghetns.sum())

            # Audit
            audits.audit_3_2_6(self, audit_original_hhpop, ntem_hhpop_trim)
            if self.export_process_files:
                compress.write_out(ntem_hhpop_trim, self.pop_trim_with_full_dims_path)

            self._logger.info('Step 3.2.6 completed. Continuing running this function as Step 3.2.7.')

            # Further adjust detailed dimensional population according to zonal dwelling type from crp
            # This is Section 3.2.7 (to the end of this function)
            # Title of section 3.2.7 is "Verify population profile by dwelling type"
            # TODO ART, update script around here to reflect Jupyter - Done? It seems to work anyway...
            #  still need to add audits for 3.2.6 above this though

            normits_hhpop_bydt = aj_crp.rename(columns=const.NORMITS_HHPOP_BYDT_COL_RENAME)
            ntem_hhpop_bydt = ntem_hhpop_trim.groupby(['z', 't'])['P_aghetns'].sum().reset_index()
            ntem_hhpop_bydt = ntem_hhpop_bydt.rename(columns=const.NTEM_HHPOP_BYDT_COL_RENAME)
            if self.export_process_files:
                gen.safe_dataframe_to_csv(ntem_hhpop_bydt, self.ntem_hhpop_bydt_path, index=False)

            # Testing with Manchester
            # NTEM_HHpop_byDt_total_E02001045 = NTEM_HHpop_byDt[NTEM_HHpop_byDt['z'] == '1013']
            # NTEM_HHpop_byDt_total_E02001045.to_csv('NTEM_HHpop_byDt_total_E02001045.csv', index=False)

            # TODO ART, 04/02/2022: Change these variable names to be gb instead of uk (Done NK).
            #  UK includes Northern Ireland and these variables do not.
            gb_ave_hh_occ = normits_hhpop_bydt.copy()
            gb_ave_hh_occ['pop_pre_aj'] = gb_ave_hh_occ['crp_P_t'] / gb_ave_hh_occ['pop_aj_factor']
            gb_ave_hh_occ = gb_ave_hh_occ.groupby(['t'])[['properties', 'pop_pre_aj']].sum()
            gb_ave_hh_occ['UK_average_hhocc'] = gb_ave_hh_occ['pop_pre_aj'] / gb_ave_hh_occ['properties']

            hhpop = ntem_hhpop_trim.merge(ntem_hhpop_bydt, how='left', on=['z', 't'])
            # Where the problem occur: Does it still occur?
            hhpop = hhpop.merge(normits_hhpop_bydt,
                                how='left',
                                left_on=['msoa11cd', 't'],
                                right_on=['MSOA', 't']).drop(
                columns={'MSOA', 'pop_aj_factor', 'Zone'}).rename(
                columns={'msoa11cd': 'MSOA'})

            hhpop.loc[hhpop['household_occupancy_%s' % self.base_year[-2:]].isnull(),
                      ['household_occupancy_%s' % self.base_year[-2:]]] = hhpop['t'].map(gb_ave_hh_occ.UK_average_hhocc)
            hhpop.fillna({'properties': 0, 'crp_P_t': 0}, inplace=True)

            hhpop['P_aghetns_aj_factor'] = hhpop['crp_P_t'] / hhpop['P_t']
            hhpop['P_aghetns_aj'] = hhpop['P_aghetns'] * hhpop['P_aghetns_aj_factor']

            # Testing with Manchester
            # HH_P_aghetns_aj_E02001045 = hhpop[hhpop['MSOA'] == 'E02001045']
            # HH_P_aghetns_aj_E02001045.to_csv('HH_P_aghetns_aj_E02001045.csv', index=False)

            hhpop = hhpop.rename(columns=const.HHPOP_COL_RENAME)

            self._logger.info('total of hh pop from aj_ntem: ')
            self._logger.info(hhpop.people.sum())
            self._logger.info('total of hh pop from aj_crp: ')
            self._logger.info(aj_crp.population.sum())

            # Check the outcome compare NTEM aj pop (NTEM_HH_pop) against NorMITs pop (people)
            # adjusted according to pop by dwelling type
            # Audit
            audits.audit_3_2_7(self, hhpop, audit_original_hhpop, aj_crp)

            # get 2021 LA in
            zone_2021la = gen.safe_read_csv(self.zone_2021la_path, usecols=const.ZONE_2021LA_COLS)
            hhpop = hhpop.merge(zone_2021la, how='left',
                                left_on=['z'],
                                right_on=['NorMITs Zone']).drop(columns={'NorMITs Zone'})
            hhpop = hhpop.rename(columns={'2021 LA': '2021_LA_code', '2021 LA Name': '2021_LA_Name'})
            hhpop = hhpop[const.HHPOP_OUTPUT_COLS]
            self._logger.info('Population currently {}'.format(hhpop.people.sum()))

            # Adam - DONE, we need to think about how to organise the output files per step
            # Note that the output file name is now (correctly) GB,
            # but the variable being dumped is still mislabelled as uk.

            compress.write_out(hhpop, self.pop_with_full_dims_path)

            gen.safe_dataframe_to_csv(gb_ave_hh_occ, self.gb_ave_hh_occ_path, index=False)
            # TODO(NK,AT,CS): gb_ave_hh_occ is getting saved at output folder of 3.2.6. Is that correct?

            self._logger.info('Step 3.2.7 completed (along with some file dumping for Step 3.2.6)')
            self._logger.info('Step 3.2.6/Step 3.2.7 function has completed')
            gen.print_w_toggle('Step 3.2.7 completed (along with some file dumping for Step 3.2.6)', verbose=verbose)
            gen.print_w_toggle('Step 3.2.6/Step 3.2.7 function has completed', verbose=verbose)

    def subsets_worker_nonworker(self,
                                 called_by: str,
                                 verbose: bool = True
                                 ):

        """
        Splits Population to worker and non worker population.

        Splits Population further according to their employment status
        (1 and 2 for employed or 3-5 for all the rest of people who are not employed)
        into worker or non-worker subsets and then aggregates zonal values to district.

        Parameters
        ----------
        called_by: str.
            String containing the function from where it is called.

        verbose: bool = True
            Whether to print text or not.

        Return
        ------
        if called by "LA_level_adjustment",
        seed_worker: pd.Dataframe
            Dataframe containing input seed for IPF procedure,

        hhpop_workers_la: pd.Dataframe
            Dataframe containing fully segmented worker population at LA level.

        hhpop_nwkrs_ag_la: pd.Dataframe
            Dataframe containing non worker population by age and gender at LA level.

        hhpop_non_workers_la: pd.Dataframe
            Dataframe containing fully segmented non-worker population at LA level.

        la_2021_to_z_lookup
            Dataframe containing LA and MSOA correspondence.

        if called by "adjust_zonal_workers_nonworkers",
        hhpop_workers:
            Dataframe containing fully segmented worker population.

        hhpop_non_workers:
            Dataframe containing fully segmented non worker population.
        """
        self._logger.info('Running Step 3.2.8, which has been called by ' + called_by)
        gen.print_w_toggle('Running Step 3.2.8, which has been called by ' + called_by, verbose=verbose)

        # Read in output of Step 3.2.6/3.2.7 rather than calling the function and taking the output directly.
        # This prevents chain calling from 3.2.10 all the way back to 3.2.4!
        hhpop = compress.read_in(self.pop_with_full_dims_path)

        audit_3_2_8_data = hhpop.copy()
        audit_3_2_8_data = audit_3_2_8_data.groupby(const.HHPOP_GROUPBY)[['people']].sum().reset_index()
        audit_3_2_8_data = audit_3_2_8_data.rename(columns={'people': 'HHpop'})

        hhpop_workers = hhpop.loc[(hhpop['e'] <= 2)]
        hhpop_non_workers = hhpop.loc[(hhpop['e'] > 2)]

        hhpop_workers_la = hhpop_workers.groupby(const.HHPOP_GROUPBY)[['people']].sum().reset_index()
        audit_hhpop_workers_la = hhpop_workers_la.copy()
        hhpop_non_workers_la = hhpop_non_workers.groupby(const.HHPOP_GROUPBY)[['people']].sum().reset_index()
        audit_hhpop_non_workers_la = hhpop_non_workers_la.copy()

        # check totals
        # print(HHpop.people.sum())
        # print(HHpop_workers.people.sum())
        # print(HHpop_non_workers.people.sum())
        # print(hhpop_workers_la.people.sum() + hhpop_non_workers_la.people.sum())

        # gender and employment status combined for workers to prepare for furness on LA level
        ge_combination = [(hhpop_workers_la['g'] == 2) & (hhpop_workers_la['e'] == 1),
                          (hhpop_workers_la['g'] == 2) & (hhpop_workers_la['e'] == 2),
                          (hhpop_workers_la['g'] == 3) & (hhpop_workers_la['e'] == 1),
                          (hhpop_workers_la['g'] == 3) & (hhpop_workers_la['e'] == 2)]

        hhpop_workers_la['ge'] = np.select(ge_combination, const.GE_COMBINATION_VALUES)
        seed_worker = hhpop_workers_la[const.SEED_WORKER_COLS]
        # print(hhpop_workers_la.head(5))
        # print(hhpop_workers_la.tail(5))

        hhpop_nwkrs_ag_la = hhpop_non_workers_la.groupby(['2021_LA_code', 'a', 'g'])[['people']].sum().reset_index()
        # the following outputs are just for checking purpose
        hhpop_wkrs_ge_la = hhpop_workers_la.groupby(['2021_LA_code', 'ge'])[['people']].sum().reset_index()
        hhpop_wkrs_s_la = hhpop_workers_la.groupby(['2021_LA_code', 's'])[['people']].sum().reset_index()

        self._logger.info('Worker currently {}'.format(hhpop_workers.people.sum()))
        self._logger.info('Non_worker currently {}'.format(hhpop_non_workers.people.sum()))
        self._logger.info('Population currently {}'.format(hhpop_workers.people.sum() + hhpop_non_workers.people.sum()))

        # Audit
        audits.audit_3_2_8(self, audit_3_2_8_data, audit_hhpop_workers_la, audit_hhpop_non_workers_la,
                           hhpop_workers, hhpop_non_workers)

        gen.print_w_toggle('Step 3.2.8 completed (it is just returning outputs now...)', verbose=verbose)

        # Return variables based on function calling this step
        if called_by == 'LA_level_adjustment':
            la_2021_to_z_lookup = hhpop_workers[['2021_LA_code', 'MSOA']]
            la_2021_to_z_lookup = la_2021_to_z_lookup.drop_duplicates()
            self._logger.info('Step 3.2.8 completed - Called by step 3.2.9')
            self._logger.info('Returning variable "seed_worker" in internal memory')
            self._logger.info('Returning variable "hhpop_workers_LA" in internal memory')
            self._logger.info('Returning variable "hhpop_nwkrs_ag_la" in internal memory')
            self._logger.info('Returning variable "hhpop_non_workers_LA" in internal memory')
            self._logger.info('Returning variable "la_2021_to_z_lookup" in internal memory')
            self._logger.info('Note that no files have been written out from this call')
            gen.print_w_toggle('Returned variables:', verbose=verbose)
            gen.print_w_toggle(
                'seed_worker, hhpop_workers_LA, HHpop_nwrkrs_LA, hhpop_non_workers_LA, la_2021_to_z_lookup',
                verbose=verbose)
            return [seed_worker, hhpop_workers_la, hhpop_nwkrs_ag_la, hhpop_non_workers_la, la_2021_to_z_lookup]
        elif called_by == 'adjust_zonal_workers_nonworkers':
            self._logger.info('Step 3.2.8 completed - Called by step 3.2.10')
            self._logger.info('Returning variable "hhpop_workers_la" in internal memory')
            gen.print_w_toggle('Returned variable adjust_zonal_workers_nonworkers', verbose=verbose)
            return [hhpop_workers, hhpop_non_workers]
        else:
            # Adam - DONE, we need to think how to organise the structure of outputs files per step
            # Saving files out only when called by functions outside of this .py file

            gen.safe_dataframe_to_csv(hhpop_nwkrs_ag_la, self.hhpop_nwkrs_ag_la_path, index=False)
            gen.safe_dataframe_to_csv(hhpop_wkrs_ge_la, self.hhpop_wkrs_ge_la_path, index=False)
            gen.safe_dataframe_to_csv(hhpop_wkrs_s_la, self.hhpop_wkrs_s_la_path, index=False)

            self._logger.info('Step 3.2.8 completed - not called by a 3.2.x function')
            self._logger.info('Returning only a short list stating no output was requested')
            self._logger.info('HHpop data has been saved to file though')
            gen.print_w_toggle('Was not called by a valid 3.2.x function, no data returned, but saved HHpop to file',
                               verbose=verbose)
            return pd.DataFrame([], columns=['No data requested', 'but data saved to file'])

    def la_level_adjustment(self,
                            verbose: bool = True
                            ) -> None:
        """
        Examines employment status (fte/pte), gender (male/female) and SOC
        (higher, medium and lower) profile of workers at district level
        according to available data of Annual Population Survey (APS) and MYPE
        from NOMIS for base year (2018 or 2019); and adjusts age and gender
        profile for non-workers at district level using MYPE for base year (2018 or 2019).
        APS data is a residence-based labour market survey published at local authority level
        or above focusing on residents’ employment status. It provides decent level of details
        on the profile of gender and fte/pte status and SOC of employed residents and this is
        the main driving force to have worker and non-worker separated from total household
        population and to have each set of data verified at district level.


        Parameters
        ----------
        verbose: bool = True
            Whether to print text or not.

        Return
        ------
        None
        """

        self._logger.info('Running Step 3.2.9')
        gen.print_w_toggle('Running Step 3.2.9', verbose=verbose)
        la_level_adjustment_name = 'LA_level_adjustment'
        # Adam - DONE, the inputs here should be called in from the outcome of your scripts from function MYE_APS_process
        # Added a manual control to allow read in from file (rather than internal memory) in the event of a partial run
        # False means "read in from memory" and is the default.
        read_base_year_pop_msoa_path_file = False

        if read_base_year_pop_msoa_path_file:

            la_id = pd.read_csv(self.la_info_for_2021_path)
            la_worker_control = gen.safe_read_csv(self.hhr_worker_by_d_for_export_path,
                                                  usecols=['2021_LA_name', '2021_LA_code', 'LA', 'Worker'])
            la_worker_ge_control = gen.safe_read_csv(self.hhr_worker_by_d_for_export_path,
                                                     usecols=['2021_LA_name', '2021_LA_code', 'LA', 'Male fte',
                                                              'Male pte',
                                                              'Female fte', 'Female pte'])
            la_worker_s_control = gen.safe_read_csv(self.hhr_worker_by_d_for_export_path,
                                                    usecols=['2021_LA_name', '2021_LA_code', 'LA', 'higher', 'medium',
                                                             'skilled'])
            la_nonworker_control = gen.safe_read_csv(self.hhr_nonworker_by_d_for_export_path,
                                                     usecols=['2021_LA_name', '2021_LA_code', 'LA', 'Non worker'])
            la_nonworker_ag_control = gen.safe_read_csv(self.hhr_nonworker_by_d_for_export_path,
                                                        usecols=['2021_LA_name', '2021_LA_code', 'LA', 'Children',
                                                                 'M_16-74', 'F_16-74', 'M_75 and over',
                                                                 'F_75 and over'])

            pe_dag_for_audit = gen.safe_read_csv(self.hhr_pop_by_dag)

            self._logger.info('Step 3.2.9 read in data processed by the MYE_APS_process through an existing csv')
            self._logger.info('WARNING - This is not the default way of reading this data!')
            self._logger.info('Did you mean to do that?')
        else:
            self._logger.info(
                'Step 3.2.9 is calling the MYE_APS_process function in order to obtain Base Year population data')

            la_worker_df_import, la_nonworker_df_import, la_id, pe_dag_for_audit = self.mye_aps_process(
                la_level_adjustment_name)

            self._logger.info('Step 3.2.9 read in data processed by the MYE_APS_process function from internal memory')

            la_worker_control = la_worker_df_import[['2021_LA_name', '2021_LA_code', 'LA', 'Worker']]
            la_worker_ge_control = la_worker_df_import[[
                '2021_LA_name', '2021_LA_code', 'LA', 'Male fte', 'Male pte', 'Female fte', 'Female pte']]
            la_worker_s_control = la_worker_df_import[[
                '2021_LA_name', '2021_LA_code', 'LA', 'higher', 'medium', 'skilled']]
            la_nonworker_control = la_nonworker_df_import[[
                '2021_LA_name', '2021_LA_code', 'LA', 'Non worker']]
            la_nonworker_ag_control = la_nonworker_df_import[[
                '2021_LA_name', '2021_LA_code', 'LA', 'Children', 'M_16-74', 'F_16-74', 'M_75 and over',
                'F_75 and over']]

        # Call the file containing Pe from step 3.2.5's outputs

        pe_df = gen.safe_read_csv(self.hhr_vs_all_pop_path)

        # seed_worker not exported as a file, so wll always need to read directly from a function 3.2.8 call
        # Whilst reading it in, can also read in HHpop variables
        seed_worker, hhpop_workers_la, hhpop_nwkrs_ag_la, \
        hhpop_non_workers_d, z_2_la = self.subsets_worker_nonworker(called_by=la_level_adjustment_name, verbose=verbose)

        pe_df = pe_df[['MSOA', 'Total_HHR']]
        pe_df = pd.merge(pe_df, z_2_la, how='left', on='MSOA')
        pe_df = pe_df.groupby('2021_LA_code')['Total_HHR'].sum().reset_index()

        la_nonworker_control = la_nonworker_control.rename(columns={'Non worker': 'nonworker'})
        la_worker_ge_control = pd.melt(la_worker_ge_control,
                                       id_vars=['2021_LA_name', '2021_LA_code', 'LA'],
                                       value_vars=['Male fte', 'Male pte', 'Female fte',
                                                   'Female pte']).rename(columns={'variable': 'ge', 'value': 'worker'})
        # print(LA_worker_ge_control.head(5))
        la_worker_s_control = pd.melt(la_worker_s_control,
                                      id_vars=['2021_LA_name', '2021_LA_code', 'LA'],
                                      value_vars=['higher', 'medium',
                                                  'skilled']).rename(columns={'variable': 's', 'value': 'worker'})
        # print(LA_worker_s_control.head(5))
        la_nonworker_ag_control = pd.melt(la_nonworker_ag_control,
                                          id_vars=['2021_LA_name', '2021_LA_code', 'LA'],
                                          value_vars=['Children', 'M_16-74', 'F_16-74', 'M_75 and over',
                                                      'F_75 and over']).rename(
            columns={'variable': 'ag', 'value': 'nonworker'})
        # print(LA_nonworker_ag_control.head(5))
        # print('\n')
        self._logger.info('number of worker (by ge): ')
        self._logger.info(la_worker_ge_control.worker.sum())
        self._logger.info('number of worker (by s): ')
        self._logger.info(la_worker_s_control.worker.sum())
        self._logger.info('number of nonworker: ')
        self._logger.info(la_nonworker_ag_control.nonworker.sum())
        self._logger.info('number of hh pop: ')
        self._logger.info(la_worker_control.Worker.sum() + la_nonworker_control.nonworker.sum())

        la_worker_ge_control['ge'] = la_worker_ge_control['ge'].map(const.GE)
        la_worker_s_control['s'] = la_worker_s_control['s'].map(const.S)
        la_nonworker_ag_control['a'] = la_nonworker_ag_control['ag'].map(const.A)
        la_nonworker_ag_control['g'] = la_nonworker_ag_control['ag'].map(const.G)

        seed_worker = seed_worker.merge(
            la_id, how='left', on='2021_LA_code').drop(columns={'2021_LA_name', '2021_LA_code'})

        seed_worker = seed_worker.rename(columns={"LA": "d", "people": "total"})
        seed_worker = seed_worker[['d', 'ge', 's', 'a', 'h', 't', 'n', 'total']]

        ctrl_ge = la_worker_ge_control.copy()
        ctrl_ge = ctrl_ge.rename(
            columns={"LA": "d", "worker": "total"}).drop(columns={'2021_LA_name', '2021_LA_code'})

        ctrl_s = la_worker_s_control.copy()
        ctrl_s = ctrl_s.rename(
            columns={"LA": "d", "worker": "total"}).drop(columns={'2021_LA_name', '2021_LA_code'})

        # Export seed_worker, then read it back in again. Failure to do so will result in the script crashing in ipfn
        # It is unclear why this occurs.
        gen.safe_dataframe_to_csv(seed_worker, self.seed_worker_path, index=False)
        seed = gen.safe_read_csv(self.seed_worker_path)

        ctrl_ge = ctrl_ge.groupby(['d', 'ge'])['total'].sum()
        ctrl_s = ctrl_s.groupby(['d', 's'])['total'].sum()

        aggregates = [ctrl_ge, ctrl_s]
        dimensions = [['d', 'ge'], ['d', 's']]

        gen.print_w_toggle('IPFN process started', verbose=verbose)
        ipf = ipfn.ipfn(seed, aggregates, dimensions)
        seed = ipf.iteration()
        gen.print_w_toggle('IPFN process complete', verbose=verbose)

        # Following 2 lines not used, but keeping so they can be dumped for QA later if required
        # Wk_ge = seed.groupby(['d', 'ge'])['total'].sum()
        # Wk_s = seed.groupby(['d', 's'])['total'].sum()

        hhpop_workers_la = hhpop_workers_la.merge(la_id, how='left', on='2021_LA_code').drop(columns={'2021_LA_name'})
        hhpop_workers_la = hhpop_workers_la.rename(columns={'LA': 'd'})
        furnessed_worker_la = seed.copy()
        furnessed_worker_la_iterator = zip(furnessed_worker_la['d'],
                                           furnessed_worker_la['ge'],
                                           furnessed_worker_la['s'],
                                           furnessed_worker_la['a'],
                                           furnessed_worker_la['h'],
                                           furnessed_worker_la['t'],
                                           furnessed_worker_la['n'])
        furnessed_worker_la['key'] = ['_'.join([str(d), str(ge), str(s), str(a), str(h), str(t), str(n)])
                                      for d, ge, s, a, h, t, n in furnessed_worker_la_iterator]
        furnessed_worker_la = furnessed_worker_la[['key', 'total']]
        hhpop_workers_la_iterator = zip(hhpop_workers_la['d'],
                                        hhpop_workers_la['ge'],
                                        hhpop_workers_la['s'],
                                        hhpop_workers_la['a'],
                                        hhpop_workers_la['h'],
                                        hhpop_workers_la['t'],
                                        hhpop_workers_la['n'])
        hhpop_workers_la['key'] = ['_'.join([str(d), str(ge), str(s), str(a), str(h), str(t), str(n)])
                                   for d, ge, s, a, h, t, n in hhpop_workers_la_iterator]
        hhpop_workers_la = hhpop_workers_la.merge(furnessed_worker_la, how='left', on=['key'])

        hhpop_workers_la['wkr_aj_factor'] = hhpop_workers_la['total'] / hhpop_workers_la['people']
        hhpop_workers_la['wkr_aj_factor'] = hhpop_workers_la['wkr_aj_factor'].fillna(1)

        # New df called aj_hhpop_workers_la copy of
        #  hhpop_workers_la[['2021_LA_code', 'a', 'g', 'h', 'e', 't', 'n', 's', 'total']]
        aj_hhpop_workers_la = hhpop_workers_la.copy()
        aj_hhpop_workers_la = aj_hhpop_workers_la[const.AJ_HHPOP_WORKERS_LA_COLS]

        wkrs_aj_factor_la = hhpop_workers_la[const.WRKS_AJ_FACTOR_LA_COLS]
        self._logger.info('worker currently {}'.format(hhpop_workers_la.total.sum()))

        hhpop_nwkrs_ag_la = hhpop_nwkrs_ag_la.merge(la_nonworker_ag_control, how='left', on=['2021_LA_code', 'a', 'g'])
        hhpop_nwkrs_ag_la['nwkr_aj_factor'] = hhpop_nwkrs_ag_la['nonworker'] / hhpop_nwkrs_ag_la['people']

        nwkrs_aj_factor_la = hhpop_nwkrs_ag_la[const.NWKRS_AJ_FACTOR_LA]
        self._logger.info('non_worker currently {}'.format(hhpop_nwkrs_ag_la.nonworker.sum()))

        # Fetches hhpop_non_workers_la from Step 3.2.8. Merge on LA, a ,g with nwkrs_aj_factor_la to get
        #  'nwkr_aj_factor' in. Resulting df is aj_hhpop_non_workers_la with heading 'total' in it. This heading is
        #  the hhpop for nonworkers (i.e. column 'people') x 'nwkr_aj_factor'.
        aj_hhpop_non_workers_la = pd.merge(hhpop_non_workers_d, nwkrs_aj_factor_la,
                                           how='left', on=['2021_LA_code', 'a', 'g'])
        aj_hhpop_non_workers_la['total'] = aj_hhpop_non_workers_la['people'] * aj_hhpop_non_workers_la['nwkr_aj_factor']

        # Append aj_hhpop_non_workers_la to aj_hhpop_workers_la to get full district level household pop with full
        #  dimensions. Groupby on d (for 'total'). Compare to MYPE on household population. Dump file as audit. First line
        #  of final Step 3.2.9 Audit bullet point. Groupby d, a ,g  (for 'total'). Compare to MYPE household population.
        #  Dump as audit. Second line of final Step 3.2.9 bullet point. All comparisons should be very near 0.

        # Format output files to tfn tt instead of NorMITs segmentation
        seg_to_tt_df = gen.safe_read_csv(self.normits_seg_to_tfn_tt)
        aj_hhpop_workers_la_out = aj_hhpop_workers_la.merge(seg_to_tt_df, on=const.SEG_TO_TT_COLS)
        aj_hhpop_workers_la_out = aj_hhpop_workers_la_out[['2021_LA_code', 'tfn_tt', 't', 'total']]
        aj_hhpop_non_workers_la_out = aj_hhpop_non_workers_la.merge(seg_to_tt_df, on=const.SEG_TO_TT_COLS)
        aj_hhpop_non_workers_la_out = aj_hhpop_non_workers_la_out[
            ['2021_LA_code', 'tfn_tt', 't', 'people', 'nwkr_aj_factor', 'total']]

        # Audit
        audits.audit_3_2_9(self,
                           audit_hhpop_by_d=aj_hhpop_workers_la,
                           aj_hhpop_non_workers_la=aj_hhpop_non_workers_la,
                           pe_df=pe_df,
                           pe_dag_for_audit=pe_dag_for_audit,
                           hhpop_workers_la=hhpop_workers_la,
                           hhpop_nwkrs_ag_la=hhpop_nwkrs_ag_la,
                           )

        # Export files
        gen.safe_dataframe_to_csv(seed, self.seed_path, index=False)
        gen.safe_dataframe_to_csv(nwkrs_aj_factor_la, self.nwkrs_aj_factor_la_path, index=False)
        gen.safe_dataframe_to_csv(wkrs_aj_factor_la, self.wkrs_aj_factor_la_path, index=False)
        compress.write_out(aj_hhpop_workers_la_out, self.verified_d_worker_path)
        compress.write_out(aj_hhpop_non_workers_la_out, self.verified_d_non_worker_path)

        self._logger.info('Step 3.2.9 completed')
        gen.print_w_toggle('Step 3.2.9 completed', verbose=verbose)

    def adjust_zonal_workers_nonworkers(self,
                                        zonal_properties_by_t: pd.DataFrame,
                                        verbose: bool = True
                                        ) -> None:
        """
        Adjusts zonal household population according to verified values on workers
        and non-workers per district derived from previous step.

        Parameters
        ----------
        zonal_properties_by_t: pd.DataFrame,
            Dataframe containing MYPE compliant household population segmented
            by zone and dwelling type.

        verbose: bool = True
            Whether to print the text or not.

        Returns
        -------
        None

        """

        self._logger.info('Running Step 3.2.10')
        gen.print_w_toggle('Running Step 3.2.10', verbose=verbose)
        adjust_zonal_workers_nonworkers_name = 'adjust_zonal_workers_nonworkers'
        # Had to save a copy of the xlsx as a csv as read_excel is annoying and doesn't support xlsx files anymore!
        nomis_mye_base_year = gen.safe_read_csv(self.nomis_mye_base_year_path, skiprows=6)
        nomis_mye_base_year = nomis_mye_base_year[
            ['local authority: district / unitary (as of April 2021)', 'All Ages']]
        nomis_mye_base_year = nomis_mye_base_year.rename(
            columns={'local authority: district / unitary (as of April 2021)': '2021_LA_Name',
                     'All Ages': 'MYE_pop'})

        # Call function for 3.2.8 to get hhpop_workers
        # (because 3.2.8 is a quick to run function vs the file read/write time)

        # Read wkrs_aj_factor_LA from csv dumped by 3.2.9 (because IPFN can take a long time to run)

        hhpop_workers, hhpop_non_workers = self.subsets_worker_nonworker(
            adjust_zonal_workers_nonworkers_name,
            verbose=verbose)

        # Read wkrs_aj_factor_LA from csv dumped by 3.2.9 (because IPFN can take a long time to run)
        wkrs_aj_factor_la = gen.safe_read_csv(self.wkrs_aj_factor_la_path)
        nwkrs_aj_factor_la = gen.safe_read_csv(self.nwkrs_aj_factor_la_path)

        # Read average uk hh occupancy from step 3.2.6
        uk_ave_hh_occ_lookup = gen.safe_read_csv(self.gb_ave_hh_occ_path)

        hhpop_workers = hhpop_workers.merge(wkrs_aj_factor_la,
                                            how='left',
                                            on=const.HHPOP_GROUPBY)
        hhpop_workers['aj_worker'] = hhpop_workers['people'] * hhpop_workers['wkr_aj_factor']
        #   gen.print_w_toggle(hhpop_workers.head(5), verbose=verbose)
        #   gen.print_w_toggle(hhpop_workers.people.sum(), verbose=verbose)
        #   gen.print_w_toggle(hhpop_workers.aj_worker.sum(), verbose=verbose)
        hhpop_workers = hhpop_workers.drop(columns={'people'})
        hhpop_workers = hhpop_workers.rename(columns={'aj_worker': 'people', 'wkr_aj_factor': 'scaling_factor'})

        hhpop_non_workers = hhpop_non_workers.merge(nwkrs_aj_factor_la, how='left', on=['2021_LA_code', 'a', 'g'])
        hhpop_non_workers['aj_nonworker'] = hhpop_non_workers['people'] * hhpop_non_workers['nwkr_aj_factor']
        #   gen.print_w_toggle(hhpop_non_workers.head(5))
        #   gen.print_w_toggle(hhpop_non_workers.people.sum())
        #   gen.print_w_toggle(hhpop_non_workers.aj_nonworker.sum())
        hhpop_non_workers = hhpop_non_workers.drop(columns={'people'})
        hhpop_non_workers = hhpop_non_workers.rename(
            columns={'aj_nonworker': 'people', 'nwkr_aj_factor': 'scaling_factor'})

        hhpop_combined = hhpop_non_workers.copy()
        hhpop_combined = hhpop_combined.append(hhpop_workers, ignore_index=True)
        hhpop_combined = hhpop_combined.drop(columns=['properties', 'scaling_factor'])

        final_zonal_hh_pop_by_t = hhpop_combined.groupby(['MSOA', 'z', 't'])['people'].sum()
        final_zonal_hh_pop_by_t = final_zonal_hh_pop_by_t.reset_index()
        final_zonal_hh_pop_by_t_iterator = zip(final_zonal_hh_pop_by_t['z'], final_zonal_hh_pop_by_t['t'])
        final_zonal_hh_pop_by_t['z_t'] = ['_'.join([str(z), str(t)]) for z, t in final_zonal_hh_pop_by_t_iterator]

        # zonal_properties_by_t = gen.safe_read_csv(self.mye_pop_compiled_path)

        zonal_properties_by_t_iterator = zip(zonal_properties_by_t['Zone'],
                                             zonal_properties_by_t['census_property_type'])
        zonal_properties_by_t['z_t'] = ['_'.join([str(z), str(t)]) for z, t in zonal_properties_by_t_iterator]

        final_zonal_hh_pop_by_t = pd.merge(final_zonal_hh_pop_by_t,
                                           zonal_properties_by_t,
                                           how='left',
                                           on='z_t')
        final_zonal_hh_pop_by_t['final_hh_occ'] = final_zonal_hh_pop_by_t['people'] / final_zonal_hh_pop_by_t['UPRN']
        final_zonal_hh_pop_by_t.loc[
            final_zonal_hh_pop_by_t['final_hh_occ'].isnull(),
            'final_hh_occ'] = final_zonal_hh_pop_by_t['t'].map(uk_ave_hh_occ_lookup.UK_average_hhocc)
        final_zonal_hh_pop_by_t['UPRN'].fillna(0, inplace=True)
        final_zonal_hh_pop_by_t.rename(columns={'UPRN': 'Properties'}, inplace=True)
        final_zonal_hh_pop_by_t = final_zonal_hh_pop_by_t[['MSOA', 'z', 't', 'people', 'Properties', 'final_hh_occ']]

        # Check zonal total vs MYE HHpop at zonal level min, max, mean
        self._logger.info('Total hhpop is now:')
        self._logger.info(hhpop_combined.people.sum())

        mye_msoa_pop = gen.safe_read_csv(self.mye_msoa_pop)

        mye_hhr_pop = mye_msoa_pop[['MSOA', 'Total_HHR']]
        self._logger.info('Original hhpop is:')
        self._logger.info(mye_hhr_pop.Total_HHR.sum())

        hhpop_combined_check_z = hhpop_combined.copy()
        hhpop_combined_check_z = hhpop_combined_check_z[['MSOA', 'people']]
        hhpop_combined_check_z = hhpop_combined_check_z.groupby(['MSOA']).sum()
        hhpop_combined_check_z = hhpop_combined_check_z.merge(mye_hhr_pop, how='outer', on='MSOA')

        hhpop_combined_check_la = hhpop_combined_check_z.copy()

        hhpop_combined_check_z['percentage_diff'] = (hhpop_combined_check_z['people'] / hhpop_combined_check_z[
            'Total_HHR']) - 1
        self._logger.info('Check zonal level totals:')
        self._logger.info('The min %age diff is {}%'.format(str(hhpop_combined_check_z['percentage_diff'].min() * 100)))
        self._logger.info('The max %age diff is {}%'.format(str(hhpop_combined_check_z['percentage_diff'].max() * 100)))
        self._logger.info('The mean %age diff is {}%'.format(str(hhpop_combined_check_z['percentage_diff'].mean()
                                                                 * 100)))

        hhpop_combined_pdiff_min = hhpop_combined_check_z.loc[
            hhpop_combined_check_z.percentage_diff.idxmin()]
        hhpop_combined_pdiff_max = hhpop_combined_check_z.loc[
            hhpop_combined_check_z.percentage_diff.idxmax()]
        hhpop_combined_pdiff_min = pd.DataFrame(hhpop_combined_pdiff_min).transpose()
        hhpop_combined_pdiff_max = pd.DataFrame(hhpop_combined_pdiff_max).transpose()
        hhpop_combined_pdiff_extremes = hhpop_combined_pdiff_min.append(hhpop_combined_pdiff_max)
        self._logger.info(hhpop_combined_pdiff_extremes)

        # Check LA total vs MYE HHpop - there should be 0% variance
        la_2_z = hhpop_combined.copy()
        la_2_z = la_2_z[['2021_LA_Name', 'MSOA']].drop_duplicates().reset_index().drop(columns=['index'])
        hhpop_combined_check_la = pd.merge(hhpop_combined_check_la, la_2_z, how='left', on='MSOA')
        hhpop_combined_check_la = hhpop_combined_check_la.drop(columns=['MSOA'])
        hhpop_combined_check_la = hhpop_combined_check_la.groupby(['2021_LA_Name']).sum()
        hhpop_combined_check_la['percentage_diff'] = (hhpop_combined_check_la['people'] / hhpop_combined_check_la[
            'Total_HHR']) - 1
        self._logger.info('Check district level totals:')
        self._logger.info('The min %age diff is {}%'.format(str(hhpop_combined_check_la['percentage_diff'].min()
                                                                * 100)))
        self._logger.info('The max %age diff is {}%'.format(str(hhpop_combined_check_la['percentage_diff'].max()
                                                                * 100)))
        self._logger.info('The mean %age diff is {}%'.format(str(hhpop_combined_check_la['percentage_diff'].mean()
                                                                 * 100)))

        # Audit
        audits.audit_3_2_10(self, hhpop_combined_check_z, hhpop_combined_check_la, hhpop_combined,
                            hhpop_combined_pdiff_extremes)

        # Now call 3.2.11 directly from 3.2.10.
        # This allows 3.2.10 to pass 3.2.11 to big main df directly and
        # read the output back in and merge it to create the final output.
        self._logger.info('Step 3.2.10 is calling step 3.2.11 to generate CER pop data')
        gen.print_w_toggle('Step 3.2.10 is calling Step 3.2.11', verbose=verbose)
        expanded_cer_pop = self.process_cer_data(hhpop_combined, la_2_z, verbose)

        self._logger.info('Step 3.2.11 completed, returning to step 3.2.10')
        gen.print_w_toggle('Step 3.2.10 has completed its call of Step 3.2.11', verbose=verbose)

        # Append CER to the end of the HHpop table (with CER t=8)
        # So you have zaghetns population (i.e. no A - already dropped)
        # Dump this to compressed file in the pycharm script
        all_pop = hhpop_combined.copy()
        all_pop = all_pop.drop(columns=['2021_LA_code'])
        cer_pop_expanded = expanded_cer_pop.rename(columns={'Zone': 'z', 'zaghetns_CER': 'people'})
        all_pop = all_pop.append(cer_pop_expanded)

        all_pop_by_d_groupby_cols = ['2021_LA_Name']
        all_pop_by_d = all_pop.groupby(all_pop_by_d_groupby_cols)['people'].sum().reset_index()
        self._logger.info('Total pop (including CER) now:')
        self._logger.info(all_pop_by_d.people.sum())

        # Check LA level pop against MYE
        check_all_pop_by_d = all_pop_by_d.copy()
        check_all_pop_by_d = pd.merge(check_all_pop_by_d, nomis_mye_base_year, on='2021_LA_Name', how='left')
        # Remove all the pesky 1000 separators from the string interpreted numerical columns!
        check_all_pop_by_d.replace(',', '', regex=True, inplace=True)
        check_all_pop_by_d['MYE_pop'] = check_all_pop_by_d['MYE_pop'].astype(int)
        check_all_pop_by_d['pop_deviation'] = (check_all_pop_by_d['people'] / check_all_pop_by_d['MYE_pop']) - 1
        self._logger.info('The min %age diff is {}%'.format(str(check_all_pop_by_d['pop_deviation'].min() * 100)))
        self._logger.info('The max %age diff is {}%'.format(str(check_all_pop_by_d['pop_deviation'].max() * 100)))
        self._logger.info('The mean %age diff is {}%'.format(str(check_all_pop_by_d['pop_deviation'].mean() * 100)))
        self._logger.info('The overall deviation is %s people' % str(
            check_all_pop_by_d['people'].sum() - check_all_pop_by_d['MYE_pop'].sum()))

        gen.safe_dataframe_to_csv(check_all_pop_by_d, self.check_all_pop_by_d_path, index=False)

        # Also groupby this output by removing t to get zaghens population.
        # Dump this to compressed file in the pycharm script
        all_pop_by_t_groupby_cols = all_pop.columns.values.tolist()
        all_pop_by_t_groupby_cols = [x for x in all_pop_by_t_groupby_cols if x != 't' and x != 'people']
        all_pop_by_t = all_pop.groupby(all_pop_by_t_groupby_cols)['people'].sum().reset_index()

        # Audit
        audits.audit_3_2_11_10(self, all_pop_by_d, check_all_pop_by_d)

        # Auditing the bit of Step 3.2.11 that is carried out directly in the Step 3.2.10 function
        # Format ouputs
        seg_to_tt_df = pd.read_csv(self.normits_seg_to_tfn_tt)
        hhpop_combined_out = hhpop_combined.merge(seg_to_tt_df, on=['a', 'g', 'h', 'e', 'n', 's'])
        hhpop_combined_out = hhpop_combined_out[['2021_LA_code', '2021_LA_Name', 'z', 'MSOA', 'tfn_tt', 't', 'people']]
        all_pop_out = all_pop.merge(seg_to_tt_df, on=['a', 'g', 'h', 'e', 'n', 's'])
        all_pop_out = all_pop_out[['2021_LA_Name', 'z', 'MSOA', 'tfn_tt', 't', 'people']]
        all_pop_by_t_out = all_pop_by_t.merge(seg_to_tt_df, on=['a', 'g', 'h', 'e', 'n', 's'])
        all_pop_by_t_out = all_pop_by_t_out[['2021_LA_Name', 'z', 'MSOA', 'tfn_tt', 'people']]

        # Dump outputs
        self._logger.info('Dumping final outputs...')
        compress.write_out(hhpop_combined_out, self.hhpop_combined_path)
        self._logger.info('HH pop dumped')
        gen.safe_dataframe_to_csv(final_zonal_hh_pop_by_t, self.final_zonal_hh_pop_by_t_fname)
        self._logger.info('HH pop by property type dumped')
        compress.write_out(all_pop_out, self.all_pop_path)
        self._logger.info('Total pop (by z, a, g, h, e, t, n, s) dumped')
        compress.write_out(all_pop_by_t_out, self.all_pop_by_t_path)
        self._logger.info('Total pop (by z, a, g, h, e, n, s) dumped')

        self._logger.info('Step 3.2.10 completed')
        self._logger.info('If undertaking a full run through of the Base Year LU process,')
        self._logger.info('then this should have been the last function to run.')
        self._logger.info('So the %s Base Year is DONE!' % self.base_year)
        gen.print_w_toggle('Step 3.2.10 completed', verbose=verbose)
        gen.print_w_toggle('So, in theory, all steps have been run (Step 3.2.11 via Step 3.2.10), so we are',
                           verbose=verbose)
        gen.print_w_toggle('DONE!', verbose=verbose)

    def process_cer_data(self,
                         hhpop_combined_from_3_2_10: pd.DataFrame,
                         la_2_z_from_3_2_10: pd.DataFrame,
                         verbose: bool = True,
                         ) -> pd.DataFrame:
        """
        Processes CER data and further segments them to be consistent with household
        population to have dimension of a, g, h, e, n and s using GB household population
        profile with t being type 8 which is a unique code assigned for communal establishments.

        Parameters
        ----------
        hhpop_combined_from_3_2_10: pd.DataFrame
            Dataframe containing household population from previous step.

        la_2_z_from_3_2_10: pd.DataFrame
            LA to MSOA correspondence.

        verbose: bool = True
            Whether to print text or not.

        Return
        ------
        cer_pop_expanded: pd.Dataframe
            Communal Establishment Residents by zone and TfN traveller type (z, a, g ,h, e, n, s)
        """

        # This function should ONLY be called by 3.2.10
        self._logger.info('Running Step 3.2.11')
        gen.print_w_toggle('Running Step 3.2.11', verbose=verbose)

        # Read the total pop data direct from 3.2.10 as it is ~26 million lines
        # In which case, call la_2_z from it too
        # Read MYE_MSOA_pop from file as it is small (produced by MYE_APS_process sub-function of 3.2.5)

        # Subtract HHpop from MYE total to get CER

        mye_pop = gen.safe_read_csv(self.mye_msoa_pop)

        cer_pop = hhpop_combined_from_3_2_10.copy()
        cer_pop = cer_pop[['MSOA', 'people']]
        cer_pop = cer_pop.groupby(['MSOA']).sum()
        cer_pop = pd.merge(cer_pop, mye_pop, how='left', on='MSOA')
        cer_pop['CER_pop'] = cer_pop['Total_Pop'] - cer_pop['people']

        # Suppress negative CER values to 0
        cer_pop['CER_pop'] = np.where(cer_pop['CER_pop'] < 0, 0, cer_pop['CER_pop'])

        # Use CER as weights to distribute LA level CER
        cer_pop = pd.merge(cer_pop, la_2_z_from_3_2_10, how='left', on='MSOA')
        cer_pop_la = cer_pop.copy()
        cer_pop_la = cer_pop_la.drop(columns=['MSOA'])
        cer_pop_la = cer_pop_la.groupby(['2021_LA_Name']).sum()
        cer_pop_la['Total_CER'] = cer_pop_la['Total_Pop'] - cer_pop_la['Total_HHR']
        cer_pop_la = cer_pop_la.reset_index()
        cer_pop_la = cer_pop_la.rename(columns={'CER_pop': 'CER_weight_denom'})
        cer_pop_la = cer_pop_la[['2021_LA_Name', 'CER_weight_denom', 'Total_CER']]
        cer_pop = pd.merge(cer_pop, cer_pop_la, how='left', on='2021_LA_Name')
        cer_pop['CER_weight'] = cer_pop['CER_pop'] / cer_pop['CER_weight_denom']
        cer_pop['Zonal_CER'] = cer_pop['Total_CER'] * cer_pop['CER_weight']

        uk_ave_pop_df = hhpop_combined_from_3_2_10.copy()
        # Technically this should be done after the groupby,
        # but setting t = 8 here is still produces the desired output
        # and saves having to get a 'new' 't' column in the right place afterwards
        uk_ave_pop_df['t'] = 8
        uk_ave_pop_df = uk_ave_pop_df.groupby(['a', 'g', 'h', 'e', 't', 'n', 's'])['people'].sum()
        uk_ave_pop_df = pd.DataFrame(uk_ave_pop_df).reset_index()
        uk_total_pop = uk_ave_pop_df.people.sum()
        uk_ave_pop_df['people'] = uk_ave_pop_df['people'] / uk_total_pop
        uk_ave_pop_df.rename(columns={'people': 'aghetns_pop_prop'}, inplace=True)
        self._logger.info('a, g, h, e, t, n, s population proportions for t=8 (CER pop) should sum to 1.')
        if uk_ave_pop_df['aghetns_pop_prop'].sum() == 1:
            self._logger.info('Here they sum to 1 as expected.')
        else:
            self._logger.info('!!!! WARNING !!!!')
            self._logger.info('Proportions do not sum to 1! Instead they sum to:')
            self._logger.info(uk_ave_pop_df['aghetns_pop_prop'].sum())

        # Expand cer_pop by uk_ave_pop_df
        cer_pop_expanded = cer_pop.copy()
        uk_ave_pop_df_expander = uk_ave_pop_df.copy()
        cer_pop_expanded['key'] = 0
        uk_ave_pop_df_expander['key'] = 0
        cer_pop_expanded = pd.merge(cer_pop_expanded, uk_ave_pop_df_expander, on='key').drop(columns=['key'])
        cer_pop_expanded['zaghetns_CER'] = cer_pop_expanded['Zonal_CER'] * cer_pop_expanded['aghetns_pop_prop']
        self._logger.info('Check expanded CER pop matches zonal CER pop.')
        self._logger.info('Expanded CER pop is:')
        self._logger.info(cer_pop_expanded['zaghetns_CER'].sum())
        self._logger.info('Zonal CER pop is:')
        self._logger.info(cer_pop['Zonal_CER'].sum())
        cer_pop_expanded = cer_pop_expanded[const.CER_POP_EXPANDED_COLS]

        # Dump cer_pop_expanded to compressed file for QA purposes
        compress.write_out(cer_pop_expanded, self.cer_pop_expanded_path)

        # Audit
        audits.audit_3_2_11(self, cer_pop_expanded)

        # Flagging and self._logger

        self._logger.info('Step 3.2.11 completed')
        gen.print_w_toggle('Step 3.2.11 completed', verbose=verbose)

        return cer_pop_expanded

    # def _check_state(self,
    #                  step_key: str = None):
    #     """
    #     Check state function, in development.
    #     """
    #     if step_key in self.step_keys:
    #         check = 1  # TODO: Check run status
    #         self.status[self.step_key]['status'] = check
    #
    #     else:
    #         state_dict = dict()
    #         # TODO: Derive from another step list
    #
    #         for step in range(0, len(self.step_keys)):
    #             state_dict.update(
    #                 {self.step_keys[step]: {'desc': self.steps_descs[step],
    #                                         'status': 0}})
    #
    #         for step, dat in state_dict.iteritems():
    #             # Pull from key
    #             self.status[self.step_key]['status'] = self._check_state(step)
    #
    #     pass

    def build_by_emp(self):
        """
        """

        # TODO: Build from NTEM aligned in more detail?
        # Need to see what's available

        os.chdir(self.model_folder)
        file_ops.create_folder(self.iteration, ch_dir=True)

        employment.get_emp_data(self)

        employment.skill_weight_e_cats(self)

        employment.unemp_infill(self)

        self.emp_out.to_csv(self.out_paths['emp_write_path'],
                            index=False)
