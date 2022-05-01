import numpy as np
import matplotlib.pyplot as plt


time = time = np.linspace(0, 4394, 4394)[:3305] / 110
tr_target = np.load('plot_data/target.npy')
tr_control = np.load('plot_data/control.npy')
tr_est = np.load('plot_data/estimation.npy')
error = np.load('plot_data/error.npy')
tr_target_2 = np.load('plot_data/target2.npy')
tr_est_2 = np.load('plot_data/estimation2.npy')
wdiff1 = np.load('plot_data/wdiff1.npy').reshape(-1 ,2)
wdiff2 = np.load('plot_data/wdiff2.npy').reshape(-1 ,2)



fig, axs = plt.subplots(3, 2, figsize=(16, 8))

axs[0, 0].plot(time,
        tr_control,
        color='tab:blue',
        lw='2')

axs[1, 0].scatter(time,
        tr_target,
        linestyle='solid',
        color=(165/255, 172/255, 175/255),
        s=4.5,
        zorder=100)

axs[1, 0].plot(time,
        tr_est,
        color='tab:blue',
        lw='2')

axs[0, 1].plot(time,
        error,
        color='tab:blue',
        lw='2')

axs[1, 1].scatter(time,
        tr_target_2,
        linestyle='solid',
        color=(165/255, 172/255, 175/255),
        s=4.5,
        zorder=100)

axs[1, 1].plot(time,
        tr_est_2,
        color='tab:blue',
        lw='2')

axs[2, 0].plot(time[:-2],
        np.abs(wdiff1[:, 0][:3303]),
        color=(165/255, 172/255, 175/255),
        lw='2')

axs[2, 1].plot(time[:-2],
        np.abs(wdiff2[:, 1][:3303]),
        color='tab:blue',
        lw='2')

axs[0, 0].set_xlabel('Time (s)', fontsize=12)
axs[0, 1].set_xlabel('Time (s)', fontsize=12)
axs[1, 1].set_xlabel('Time (s)', fontsize=12)
axs[1, 0].set_xlabel('Time (s)', fontsize=12)
axs[2, 0].set_xlabel('Time (s)', fontsize=12)
axs[2, 1].set_xlabel('Time (s)', fontsize=12)

axs[0, 0].set_ylabel("Angle of head rotation (°)", fontsize=12)
axs[1, 0].set_ylabel("Angle of eye rotation 1 (°)", fontsize=12)
axs[1, 1].set_ylabel("Angle of eye rotation 2 (°)", fontsize=12)
axs[0, 1].set_ylabel('Error (°)', fontsize=12)
axs[2, 0].set_ylabel('Dynamics of changes in weights 1', fontsize=12)
axs[2, 1].set_ylabel('Dynamics of changes in weights 2', fontsize=12)


axs[1, 0].legend(['Identification', 'Experimental data'])
axs[1, 1].legend(['Identification', 'Experimental data'])
axs[2, 0].legend(['W_1'])
axs[2, 1].legend(['W_2'])

# axs[0, 0].text(-0.1, 1., 'A', transform=axs[0, 0].transAxes,
#                 size=20, weight='bold')
# axs[0, 1].text(-0.075, 1., 'B', transform=axs[0, 1].transAxes,
#                 size=20, weight='bold')
# axs[1, 1].text(-0.075, 1., 'D', transform=axs[1, 1].transAxes,
#                 size=20, weight='bold')
# axs[1, 0].text(-0.1, 1., 'C', transform=axs[1, 0].transAxes,
#                 size=20, weight='bold')

plt.tight_layout()
plt.savefig('articl_plot_{}.png'.format(0))
