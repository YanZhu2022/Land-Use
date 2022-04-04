from land_use.base_land_use import census_lu
from land_use.base_land_use import by_lu


def main():
    run_census = False
    run_pop = True
    run_emp = False
    verbose = True
    export_process_files = False

    iteration = 'trial_run_NK_2'
    census_year = '2011'
    base_year = '2018'
    # Enter the NorCOM status either "import from NorCOM" or "export to NorCOM"
    norcom_status = 'import from NorCOM'

    print('Building lu run, %s' % iteration)
    print('Census year is %s' % census_year)
    print('Base year is %s' % base_year)

    if run_census:
        census_run = census_lu.CensusYearLandUse(iteration=iteration)
        census_run.build_by_pop()

    lu_run = by_lu.BaseYearLandUse(norcom_status=norcom_status,
                                   iteration=iteration,
                                   base_year=base_year,
                                   census_year=census_year,
                                   export_process_files=export_process_files,)

    if run_pop:
        lu_run.build_by_pop(verbose=verbose)

    if run_emp:
        lu_run.build_by_emp()

    return 0


if __name__ == '__main__':
    main()
