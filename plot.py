import numpy as np
import matplotlib.pyplot as plt


time = np.fromfile('plot_data/timeline.dmp', sep=';')
tr_target = np.fromfile('plot_data/target.dmp', sep=';')
tr_control = np.fromfile('plot_data/control.dmp', sep=';').reshape(-1, 2)
tr_est = np.fromfile('plot_data/estimation.dmp', sep=';')
error = np.fromfile('plot_data/error.dmp', sep=';')

fig, axs = plt.subplots(2, 2, figsize=(16, 8))

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

# axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.f'))
# axs[1, 1].yaxis._useMathText = True
# axs[1, 1].ticklabel_format(style='sci', axis='y')

axs[0, 0].set_xlabel('Time (s)', fontsize=12)
axs[0, 1].set_xlabel('Time (s)', fontsize=12)
axs[1, 1].set_xlabel('Time (s)', fontsize=12)
axs[1, 0].set_xlabel('Time (s)', fontsize=12)

axs[0, 0].set_ylabel("Angle of head rotation (°)", fontsize=12)
axs[1, 0].set_ylabel("Angle of eye rotation (°)", fontsize=12)
axs[0, 1].set_ylabel('Error (°)', fontsize=12)

# axs[0, 0].set_xticks(np.arange(len(time)))

axs[1, 0].legend(['Identification', 'Experimental data'])
axs[1, 1].legend(['W_1', 'W_2'])

axs[0, 0].text(-0.1, 1., 'A', transform=axs[0, 0].transAxes,
                size=20, weight='bold')
axs[0, 1].text(-0.075, 1., 'B', transform=axs[0, 1].transAxes,
                size=20, weight='bold')
axs[1, 1].text(-0.075, 1., 'D', transform=axs[1, 1].transAxes,
                size=20, weight='bold')
axs[1, 0].text(-0.1, 1., 'C', transform=axs[1, 0].transAxes,
                size=20, weight='bold')

axs[1, 0].set_ylim([-2, 14])

# axs[0, 0].axvline(x=[49 / 110],
#                     ymin=0.97,
#                     ymax=0.994838,
#                     c=(65/255, 65/255, 65/255),
#                     linewidth=2,
#                     zorder=0,
#                     linestyle=(0, (5, 10)),
#                     clip_on=False)

# axs[1, 0].axvline(x=[49 / 110],
#                     ymin=0.0,
#                     ymax=2.2,
#                     c=(65/255, 65/255, 65/255),
#                     linewidth=2,
#                     zorder=100,
#                     linestyle=(0, (5, 10)),
#                     clip_on=False)

# axs[0, 0].axvline(x=[1290.8 / 110],
#                     ymin=0.97,
#                     ymax=0.994838,
#                     c=(65/255, 65/255, 65/255),
#                     linewidth=2,
#                     zorder=0,
#                     linestyle=(0, (5, 10)),
#                     clip_on=False)

# axs[1, 0].axvline(x=[1290.8 / 110],
#                     ymin=0.0,
#                     ymax=2.2,
#                     c=(65/255, 65/255, 65/255),
#                     linewidth=2,
#                     zorder=100,
#                     linestyle=(0, (5, 10)),
#                     clip_on=False)

plt.tight_layout()
plt.savefig('articl_plot_{}.png'.format(0))
