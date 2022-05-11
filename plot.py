import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
        "text.usetex" : True,
        "font.family" : "Helvetica"
})
plt.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc}\usepackage[T2A]{fontenc}\usepackage[russian]{babel}'

time = np.linspace(0, 4394, 4394)[:3304] / 110
tr_target = np.load('plot_data/target.npy')
tr_control = np.load('plot_data/control.npy')
tr_est = np.load('plot_data/estimation.npy')
error = np.load('plot_data/error.npy')
tr_target_2 = np.load('plot_data/target2.npy')
tr_est_2 = np.load('plot_data/estimation2.npy')
wdiff1 = np.load('plot_data/wdiff1.npy').reshape(-1 ,2)
wdiff2 = np.load('plot_data/wdiff2.npy').reshape(-1 ,2)
neuro1 = np.squeeze(np.load('plot_data/neuro1.npy'))
neuro2 = np.load('plot_data/neuro2.npy')


fig1, axs1 = plt.subplots(2, 2, figsize=(16, 8))
fig2, axs2 = plt.subplots(2, 2, figsize=(16, 8))
# fig3, axs3 = plt.subplots(2, 2, figsize=(16, 8))
fig4, axs4 = plt.subplots(2, 1, figsize=(16, 8))

axs1[0, 0].plot(time,
        tr_control[:, 1],
        lw='2')

axs1[0, 0].plot(time,
        tr_control[:, 2],
        lw='2')

axs1[1, 0].scatter(time,
        tr_target,
        linestyle='solid',
        color=(165/255, 172/255, 175/255),
        s=4.5,
        zorder=100)

axs1[1, 0].plot(time,
        tr_est,
        color='tab:blue',
        lw='2')

axs1[0, 1].plot(time,
        error,
        color='tab:blue',
        lw='2')

axs1[1, 1].scatter(time,
        tr_target_2,
        linestyle='solid',
        color=(165/255, 172/255, 175/255),
        s=4.5,
        zorder=100)

axs1[1, 1].plot(time,
        tr_est_2,
        color='tab:blue',
        lw='2')

axs2[0, 0].plot(time[:-1],
        np.abs(wdiff1[:, 0][:3303]),
        color=(165/255, 172/255, 175/255),
        lw='2')

axs2[0, 1].plot(time[:-1],
        np.abs(wdiff2[:, 1][:3303]),
        color='tab:blue',
        lw='2')

axs2[1, 0].plot(time[:-1],
        neuro1[:3303],
        # color='tab:blue',
        lw='2')

axs2[1, 1].plot(time[:-1],
        neuro2[:3303, :, 0],
        # color='tab:orange',
        lw='2')

axs2[1, 1].plot(time[:-1],
        neuro2[:3303, :, 1],
        # color='tab:orange',
        lw='2')

axs4[0].scatter(time[:500],
        tr_target[:500],
        linestyle='solid',
        color=(165/255, 172/255, 175/255),
        s=4.5,
        zorder=100)

axs4[0].plot(time[:500],
        tr_est[:500],
        color='tab:blue',
        lw='2')

axs4[1].scatter(time[:500],
        tr_target_2[:500],
        linestyle='solid',
        color=(165/255, 172/255, 175/255),
        s=4.5,
        zorder=100)

axs4[1].plot(time[:500],
        tr_est_2[:500],
        color='tab:blue',
        lw='2')

axs4[0].plot(time[0], tr_est[0], 'o', color='tab:orange', label='_nolegend_')
axs4[0].annotate(r"$\displaystyle \hat{\zeta}(0) = " + "{:.4f}".format(tr_est[0]) + "$", 
                xycoords='data',
                xy=(time[0], tr_est[0]),
                xytext=(-40, 40),
                textcoords='offset points',
                verticalalignment='bottom',
                arrowprops=dict(arrowstyle='-'),
                clip_on=True
                )

axs4[1].plot(time[0], tr_est_2[0], 'o', color='tab:orange', label='_nolegend_')
axs4[1].annotate(r"$\displaystyle \hat{\zeta}(0) = " + "{:.4f}".format(tr_est_2[0]) + "$", 
                xycoords='data',
                xy=(time[0], tr_est_2[0]),
                xytext=(-40, 40),
                textcoords='offset points',
                verticalalignment='bottom',
                arrowprops=dict(arrowstyle='-'),
                clip_on=True
                )

axs1[1, 0].set_xlabel('Время (с)', fontsize=15)
axs1[1, 1].set_xlabel('Время (с)', fontsize=15)

axs2[1, 0].set_xlabel('Время (с)', fontsize=15)
axs2[1, 1].set_xlabel('Время (с)', fontsize=15)

axs4[1].set_xlabel('Время (с)', fontsize=15)

axs1[0, 0].set_ylabel("Угловая скорость поворота головы (°/с)", fontsize=15)
axs1[1, 0].set_ylabel("Угол поворота левого глаза (°)", fontsize=15)
axs1[1, 1].set_ylabel("Угол поворота правого глаза (°)", fontsize=15)
axs1[0, 1].set_ylabel('Ошибка (°)', fontsize=15)

axs2[1, 0].set_ylabel(r'Выход $\displaystyle \phi_1$', fontsize=15)
axs2[1, 1].set_ylabel(r'Выход $\displaystyle \phi_2$', fontsize=15)

axs4[0].set_ylabel("Угол поворота левого глаза (°)", fontsize=15)
axs4[1].set_ylabel("Угол поворота правого глаза (°)", fontsize=15)

# axs2[0].set_title('Динамика изменения весов', fontsize=20)

# axs3[0].set_title('Выход нейрона', fontsize=20)


axs1[0, 0].legend(['Y', 'Z'])
axs1[1, 0].legend(['Идентификация', 'Экспериментальные данные'])
axs1[1, 1].legend(['Идентификация', 'Экспериментальные данные'])

axs4[0].legend(['Идентификация', 'Экспериментальные данные'])
axs4[1].legend(['Идентификация', 'Экспериментальные данные'])

axs2[0, 0].legend([r'$\displaystyle W_1$'])
axs2[0, 1].legend([r'$\displaystyle W_2$'])


plt.tight_layout()
fig1.savefig("articl_plot_1.png")
fig2.savefig("articl_plot_2.png")
# fig3.savefig("articl_plot_3.png")
fig4.savefig("experiment_begin.png")
