from dof6_setup import *

if __name__ == '__main__':
    for iteration in range(0, 1):
        simulation = simulacao()
        dof6 = six_dof(
            wind = simulation.wind,
            drogue_chute = simulation.drogue_chute,
            main_chute = simulation.main_chute,
            motor_missalign = simulation.motor_missalign
        )
        if simulation.drogue_chute == True:
            simulation.droguechute()
            dof6.droguechute(
                Cd = simulation.Cd_drogue,
                diameter = simulation.diameter_drogue,
                time_open = simulation.time_open
            )

        if simulation.main_chute == True:
            simulation.mainchute()
            dof6.mainchute(
                Cd = simulation.Cd_main,
                diameter = simulation.diameter_main,
                h_open = simulation.h_open
            )

        if simulation.motor_missalign == True:
            simulation.motormissalign()
            dof6.motormissalignment(
                eps = simulation.eps,
                mu = simulation.mu
            )
        #dof3.motormissalignment(
            #psi = 1,
            #eps = 0.0
        #)

        dof6.solve_ascending(
            step = simulation.step,
            tol = simulation.tol,
            plot_trajectory = simulation.plot_trajectory,
            plot_trajectory_time = simulation.plot_t_trajectory,
            save = simulation.save,
            save_name = simulation.save_name,
        )
