First, run setup.m. It will add 'lib' to the path.

There are 7 program files.

MFISTA_L1
MFISTA_L1_nonneg
MFISTA_L1_TV
MFISTA_L1_TV_nonneg
MFISTA_L1_sqTV
MFISTA_L1_sqTV_nonneg
MFISTA_TV

Choose the one appropriate for your problem.
The usage is shown if you type 'help MFISTA_**'.

The outputs of each program are x, cost, and LOOE.

x:    image output.
cost: a list of cost values. You can check if it is converging
      properly.
LOOE: estimate of leave one out error. The details are shown in
      Obuchi and Kabashima's work.

T. Obuchi and Y. Kabashima. Cross validation in LASSO and
its acceleration, Journal of Statistical Mechanics: Theory and
Experiment, Volume 2016, 053304(1-36), May. (2016)

Shiro Ikeda
30 August 2016


