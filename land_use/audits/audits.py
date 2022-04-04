"""
Created on: Fri March 25, 2022
Updated on:

Original author: Nirmal Kumar
Last update made by:
Other updates made by:

File purpose:
Module of audit functions to carry out during runs to ensure the values
being returned make sense
"""
import pandas as pd
import numpy as np
import os

# Local imports
from land_use.utils import general as gen
from land_use.utils import timing
from land_use import lu_constants as const
from land_use.utils import compress


class AuditError(gen.LandUseError):
    """
    Exception raised for errors when auditing values
    """

    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


def audit_3_2_1(self) -> None:
    """
    Runs the audit checks required for Step 3.2.1.

    Parameters
    ----------
    -

    Returns
    -------
    None
    """
    out_lines = [
        '### Audit for Step 3.2.1 ###',
        'Created: %s' % str(timing.get_datetime()),
        'Step 3.2.1 currently does nothing, so there is nothing to audit',
    ]
    with open(self.audit_3_2_1_path, 'w') as text_file:
        text_file.write('\n'.join(out_lines))


def audit_3_2_2(self) -> None:
    """
    Runs the audit checks required for Step 3.2.2.

    Parameters
    ----------
    -

    Returns
    -------
    None
    """
    out_lines = [
        '### Audit for Step 3.2.2 ###',
        'Created: %s' % str(timing.get_datetime()),
        'Step 3.2.2 currently has no audits listed, so there is nothing to audit',
    ]
    with open(self.audit_3_2_2_path, 'w') as text_file:
        text_file.write('\n'.join(out_lines))


def audit_3_2_3(self, all_res_property: pd.DataFrame) -> None:
    """
    Runs the audit checks required for Step 3.2.3.

    Parameters
    ----------
    all_res_property: pd.DataFrame
        Dataframe containing estimated household population.

    Returns
    -------
    None
    """
    arp_msoa_audit = all_res_property.groupby('ZoneID')['population'].sum().reset_index()
    gen.safe_dataframe_to_csv(arp_msoa_audit, self.arp_msoa_audit_path, index=False)
    arp_msoa_audit_total = arp_msoa_audit['population'].sum()
    out_lines = [
        '### Audit for Step 3.2.3 ###',
        'Created: %s' % str(timing.get_datetime()),
        'The total arp population is currently: %s', str(arp_msoa_audit_total),
        'A zonal breakdown of the arp population has been created here:',
        self.arp_msoa_audit_path,
    ]
    with open(self.audit_3_2_3_path, 'w') as text_file:
        text_file.write('\n'.join(out_lines))


def audit_3_2_4(self,
                crp_for_audit: pd.DataFrame,
                processed_crp_for_audit: pd.DataFrame,
                txt1: str = "",
                ) -> None:
    """
    Runs the audit checks required for Step 3.2.4.

    Parameters
    ----------
    crp_for_audit: pd.DataFrame
        Dataframe containing estimated household population in
        7 property types.

    processed_crp_for_audit: pd.DataFrame
        Dataframe containing estimated household population in
        4 property types.

    txt1: str = ""
        Dummy string used for processing.

    Returns
    -------
    None
    """
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
        'Created: %s' % str(timing.get_datetime()),
        'The total number of properties at the end of this step is: %s' % str(processed_crp_for_audit.UPRN.sum()),
        'The total population at the end of this step is: %s' % str(processed_crp_for_audit.population.sum()),
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
    """
    Runs the audit checks required for Step 3.2.5.

    Parameters
    ----------
    audit_aj_crp: pd.DataFrame
        Dataframe containing copy of MYPE compliant household
        population segmented by zone and dwelling type.

    audit_ntem_hhpop: pd.DataFrame
        Dataframe containing MYPE compliant NTEM household population
        segmented by zone, age, gender, household composition and
        employment status.

    audit_mye_msoa_pop: pd.Dataframe
        Dataframe containing zonal household population returned from
        mye_aps_process() for audit purpose.

    mye_msoa_pop: pd.Dataframe
        Dataframe containing zonal household population returned from
        mye_aps_process().

    Returns
    -------
    None
    """
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
        'Created: %s' % str(timing.get_datetime()),
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
                audit_original_hhpop: pd.DataFrame,
                ntem_hhpop_trim: pd.DataFrame,
                ) -> None:
    """
    Runs the audit checks required for Step 3.2.6.

    Parameters
    ----------
    audit_original_hhpop: pd.DataFrame
        Dataframe containing zonal household population returned from
        mye_aps_process() for audit purpose.

    ntem_hhpop_trim: pd.DataFrame
        Dataframe containing NorMITs household population segmented by
        full dimensions required by NorMITs land use tool (z, a, g, h, e, t, n, s).

    Returns
    -------
    None
    """

    audit_ntem_hhpop_trim = ntem_hhpop_trim[['msoa11cd', 'P_aghetns']]
    audit_ntem_hhpop_trim = audit_ntem_hhpop_trim.groupby(['msoa11cd'])['P_aghetns'].sum().reset_index()
    audit_original_hhpop = audit_original_hhpop[['MSOA', 'Total_HHR']]
    audit_original_hhpop = audit_original_hhpop.rename(columns={'Total_HHR': 'MYE_pop'})
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
        'Created: %s' % str(timing.get_datetime()),
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
    """
    Runs the audit checks required for Step 3.2.7.

    Parameters
    ----------
    hhpop: pd.DataFrame
        Dataframe containing household population.

    audit_original_hhpop: pd.DataFrame
        Dataframe containing zonal household population returned from
        mye_aps_process() for audit purpose.

    aj_crp: pd.DataFrame
        Dataframe containing copy of MYPE compliant household
        population segmented by zone and dwelling type.

    Returns
    -------
    None
    """
    zonaltot = hhpop.groupby(['z', 'MSOA'])[['people', 'NTEM_HH_pop']].sum().reset_index()
    zonaltot = zonaltot.rename(columns={'people': 'NorMITs_Zonal', 'NTEM_HH_pop': 'NTEM_Zonal'})
    audit_original_hhpop = audit_original_hhpop[['MSOA', 'Total_HHR']]
    audit_original_hhpop = audit_original_hhpop.rename(columns={'Total_HHR': 'MYE_pop'})
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
        'Created: %s' % str(timing.get_datetime()),
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
    """
    Runs the audit checks required for Step 3.2.8.

    Parameters
    ----------
    audit_3_2_8_data: pd.DataFrame
        Dataframe containing ajusted NorMITS household population with
        verified dwelling profile (by z,a,g,h,e,t,n,s).

    audit_hhpop_workers_la: pd.DataFrame
        Dataframe containing fully segmented worker population at LA level.

    audit_hhpop_non_workers_la: pd.DataFrame
        Dataframe containing fully segmented non worker population at LA level.

    hhpop_workers: pd.DataFrame
        Dataframe containing fully segmented worker population.

    hhpop_non_workers: pd.DataFrame
        Dataframe containing fully segmented non worker population.

    Returns
    -------
    None
    """
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
        'Created: %s' % str(timing.get_datetime()),
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
    """
    Runs the audit checks required for Step 3.2.9.

    Parameters
    ----------
    audit_hhpop_by_d: pd.DataFrame
        Dataframe containing worker verified at the district level and segmented
        by NorMITs segmentation for the base year.

    aj_hhpop_non_workers_la: pd.DataFrame
        Dataframe containing non worker verified at the district level and segmented
        by NorMITs segmentation for the base year.

    pe_df: pd.DataFrame
        Dataframe containing zonal and household population at LA level.

    pe_dag_for_audit: pd.DataFrame
        Dataframe containing district level household population segmented
        by age and gender derived from MYPE.

    hhpop_workers_la: pd.DataFrame
        Dataframe containing fully segmented worker population at LA level.

    hhpop_nwkrs_ag_la: pd.DataFrame
        Dataframe containing non worker population by age and gender at LA level.

    Returns
    -------
    None
    """
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
        'Created: %s' % str(timing.get_datetime()),
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
    """
    Runs the audit checks required for Step 3.2.10.

    Parameters
    ----------
    hhpop_combined_check_z: pd.DataFrame
        Dataframe containing MSOA comparison between MYPE and output of this step on
        total household population.

    hhpop_combined_check_la: pd.DataFrame
        Dataframe containing District l comparison between MYPE and output of this
        step on total household population.

    hhpop_combined: pd.DataFrame
        Dataframe containing worker and non worker population.

    hhpop_combined_pdiff_extremes: pd.DataFrame
        Dataframe containing Extreme deviations of zonal total household population
        between the output of this step after zonal adjustments according to district
        level verification and MYPE.

    Returns
    -------
    None
    """

    gen.safe_dataframe_to_csv(hhpop_combined_check_z, self.hhpop_combined_check_z_path, index=False)
    gen.safe_dataframe_to_csv(hhpop_combined_check_la, self.hhpop_combined_check_la_path, index=False)
    out_lines = [
        '### Audit for Step 3.2.10 ###',
        'Created: %s' % str(timing.get_datetime()),
        'The total %s population is currently: %s' % (self.base_year, str(hhpop_combined.people.sum())),
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
    """
        Runs the audit checks required for Step 3.2.11.10

        Parameters
        ----------
        all_pop_by_d: pd.DataFrame
            Dataframe containing all population including CER.

        check_all_pop_by_d: pd.DataFrame
            Dataframe containing district level comparison between MYPE and output of
            this step on total population.

        Returns
        -------
        None
    """

    out_lines = [
        '### Audit for the parts of Step 3.2.11 carried out directly by Step 3.2.10 ###',
        'Created: %s' % str(timing.get_datetime()),
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
    """
    Runs the audit checks required for Step 3.2.11.

    Parameters
    ----------
    cer_pop_expanded: pd.DataFrame
        Dataframe containing Communal Establishment Residents by zone
        and TfN traveller type (z, a, g ,h, e, n, s).

    Returns
    -------
    None
    """
    out_lines = [
        '### Audit for Step 3.2.10 ###',
        'Created: %s' % str(timing.get_datetime()),
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
