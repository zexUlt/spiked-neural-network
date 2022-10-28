import numpy as np
import matplotlib.pyplot as plt

PATH_PREFIX = "./plots/"

# plt.rcParams.update({
#         "text.usetex" : True,
#         "font.family" : "Helvetica"
# })
# plt.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc}\usepackage[T2A]{fontenc}\usepackage[russian]{babel}'

time = np.linspace(0, 4394, 4394)[:3304] / 110
tr_target = np.load('plot_data/target.npy')
tr_control = np.load('plot_data/control.npy')
tr_est = np.load('plot_data/estimation.npy')
error = np.load('plot_data/error.npy')
tr_target_2 = np.load('plot_data/target2.npy')
tr_est_2 = np.load('plot_data/estimation2.npy')
w1 = np.load('plot_data/wdiff1.npy')
w2 = np.load('plot_data/wdiff2.npy')
neuro1 = np.squeeze(np.load('plot_data/neuro1.npy'))
neuro2 = np.load('plot_data/neuro2.npy')

print(w1.shape, w2.shape)

fig1, axs1 = plt.subplots(2, 1, figsize=(16, 8))
fig2, axs2 = plt.subplots(2, 1, figsize=(16, 8))
fig3, axs3 = plt.subplots(2, 1, figsize=(16, 8))
fig4, axs4 = plt.subplots(2, 1, figsize=(16, 8))
fig5, axs5 = plt.subplots(1, 1, figsize=(16, 8))


axs1[0].scatter(time,
        tr_target,
        linestyle='solid',
        color=(165/255, 172/255, 175/255),
        s=4.5,
        zorder=100)

axs1[0].plot(time,
        tr_est,
        color='tab:blue',
        lw='2')

axs1[1].scatter(time,
        tr_target_2,
        linestyle='solid',
        color=(165/255, 172/255, 175/255),
        s=4.5,
        zorder=100)

axs1[1].plot(time,
        tr_est_2,
        color='tab:blue',
        lw='2')

axs2[0].plot(time[:-1],
        np.linalg.norm(w1, axis=(1, 2))[:3303],
        color=(165/255, 172/255, 175/255),
        lw='2')

axs3[0].plot(time[:-1],
        np.linalg.norm(w2, axis=(1, 2))[:3303],
        color='tab:blue',
        lw='2')

axs2[1].plot(time[:-1],
        neuro1[:3303],
        # color='tab:blue',
        lw='2')

axs3[1].plot(time[:-1],
        neuro2[:3303, :, 0],
        # color='tab:orange',
        lw='2')

axs3[1].plot(time[:-1],
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

axs5.plot(time,
        error,
        color='tab:blue',
        lw='2')

axs4[0].plot(time[0], tr_est[0], 'o', color='tab:orange', label='_nolegend_')
# axs4[0].annotate(r"$\displaystyle \hat{\zeta}(0) = " + "{:.4f}".format(tr_est[0]) + "$", 
#                 xycoords='data',
#                 xy=(time[0], tr_est[0]),
#                 xytext=(-40, 40),
#                 textcoords='offset points',
#                 verticalalignment='bottom',
#                 arrowprops=dict(arrowstyle='-'),
#                 clip_on=True
#                 )

axs4[1].plot(time[0], tr_est_2[0], 'o', color='tab:orange', label='_nolegend_')
# axs4[1].annotate(r"$\displaystyle \hat{\zeta}(0) = " + "{:.4f}".format(tr_est_2[0]) + "$", 
#                 xycoords='data',
#                 xy=(time[0], tr_est_2[0]),
#                 xytext=(-40, 40),
#                 textcoords='offset points',
#                 verticalalignment='bottom',
#                 arrowprops=dict(arrowstyle='-'),
#                 clip_on=True
#                 )

axs1[1].set_xlabel('Время (с)', fontsize=15)
# axs1[1, 1].set_xlabel('Время (с)', fontsize=15)

axs2[1].set_xlabel('Время (с)', fontsize=15)
axs3[1].set_xlabel('Время (с)', fontsize=15)

axs4[1].set_xlabel('Время (с)', fontsize=15)

# axs1.set_ylabel("Угловая скорость поворота головы (°/с)", fontsize=15)
axs1[0].set_ylabel("Угол поворота левого глаза (°)", fontsize=15)
axs1[1].set_ylabel("Угол поворота правого глаза (°)", fontsize=15)
# axs1.set_ylabel('Ошибка (°)', fontsize=15)

# axs2[1].set_ylabel(r'Выход $\displaystyle \phi_1$', fontsize=15)
# axs3[1].set_ylabel(r'Выход $\displaystyle \phi_2$', fontsize=15)

axs4[0].set_ylabel("Угол поворота левого глаза (°)", fontsize=15)
axs4[1].set_ylabel("Угол поворота правого глаза (°)", fontsize=15)

# axs2[0].set_title('Динамика изменения весов', fontsize=20)

# axs3[0].set_title('Выход нейрона', fontsize=20)


# axs1.legend(['Y', 'Z'])
axs1[0].legend(['Идентификация', 'Экспериментальные данные'])
axs1[1].legend(['Идентификация', 'Экспериментальные данные'])

# axs4[0].legend(['Идентификация', 'Экспериментальные данные'])
axs4[1].legend(['Идентификация', 'Экспериментальные данные'])

# axs2[0].legend([r'$\displaystyle W_1$'])
# axs3[0].legend([r'$\displaystyle W_2$'])


plt.tight_layout()
fig1.savefig(f"{PATH_PREFIX}/estimation.png")
fig2.savefig(f"{PATH_PREFIX}/w1_dyn.png")
fig3.savefig(f"{PATH_PREFIX}/w2_dyn.png")
# fig3.savefig("articl_plot_3.png")
fig4.savefig(f"{PATH_PREFIX}/experiment_begin.png")
fig5.savefig(f"{PATH_PREFIX}/error.png")
