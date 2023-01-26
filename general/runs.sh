python run.py name='quadratic' sigma=0.01 etas=[0.01] rhos=[0.001,0.01,0.1,0.5] nit=20000 nexp=5 d_quadratic=20 gpu=False

python run.py name='class_deep_nonlinear' sigma=0.01 etas=[0.01] rhos=[0.001,0.01,0.1,0.5] nit=50000 nexp=3 HIDDEN=30 gpu=False

python run.py name='teacher_student' sigma=0.01 etas=[0.001] rhos=[0.0001,0.001,0.03,0.05] nit=50000 nexp=3 teac_n_layers=1 teac_n_nodes=100 teac_linear=False stud_n_layers=1 stud_n_nodes=100 stud_linear=False noise_=2 gpu=False

python run.py name='teacher_student' sigma=0.01 etas=[0.001] rhos=[0.0001,0.001,0.03,0.05] nit=50000 nexp=3 teac_n_layers=1 teac_n_nodes=100 teac_linear=True stud_n_layers=1 stud_n_nodes=100 stud_linear=False noise_=2 gpu=False

python run.py name='teacher_student' sigma=0.001 etas=[0.01] rhos=[0.001,0.01,0.3,0.5] nit=50000 nexp=3 teac_n_layers=1 teac_n_nodes=100 teac_linear=False stud_n_layers=1 stud_n_nodes=100 stud_linear=False noise_=2 gpu=False

python run.py name='teacher_student' sigma=0.001 etas=[0.01] rhos=[0.001,0.01,0.3,0.5] nit=50000 nexp=3 teac_n_layers=1 teac_n_nodes=100 teac_linear=True stud_n_layers=1 stud_n_nodes=100 stud_linear=False noise_=2 gpu=False



























