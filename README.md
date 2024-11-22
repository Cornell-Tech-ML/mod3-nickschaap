# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: https://minitorch.github.io/

- Overview: https://minitorch.github.io/module3.html

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# Training Results

## GPU Training Results

### Training on xor dataset

```
Epoch  0  loss  6.934308921553657 correct 27
Epoch 0 took 4.594493627548218 seconds
Average time per epoch: 4.594493627548218
Epoch  10  loss  6.837571156678296 correct 27
Epoch 10 took 1.929715871810913 seconds
Average time per epoch: 2.3439862728118896
Epoch  20  loss  7.015155250070407 correct 27
Epoch 20 took 2.5859780311584473 seconds
Average time per epoch: 2.2401961712610152
Epoch  30  loss  6.9405725796098405 correct 27
Epoch 30 took 1.890604019165039 seconds
Average time per epoch: 2.1705772492193405
Epoch  40  loss  6.770979053260824 correct 27
Epoch 40 took 1.9485578536987305 seconds
Average time per epoch: 2.158791943294246
Epoch  50  loss  6.939360351893311 correct 27
Epoch 50 took 2.060309886932373 seconds
Average time per epoch: 2.1665514543944715
Epoch  60  loss  6.800175707565649 correct 27
Epoch 60 took 1.905709981918335 seconds
Average time per epoch: 2.141398132824507
Epoch  70  loss  6.43852914803912 correct 27
Epoch 70 took 1.8792400360107422 seconds
Average time per epoch: 2.131165454085444
Epoch  80  loss  6.821148719515909 correct 27
Epoch 80 took 1.8985862731933594 seconds
Average time per epoch: 2.124039093653361
Epoch  90  loss  6.482526525930806 correct 27
Epoch 90 took 2.0441770553588867 seconds
Average time per epoch: 2.1133597447321963
Epoch  100  loss  6.559551275484139 correct 27
Epoch 100 took 1.9288222789764404 seconds
Average time per epoch: 2.1095378446106863
Epoch  110  loss  6.809640671021098 correct 28
Epoch 110 took 1.8985295295715332 seconds
Average time per epoch: 2.107384531347601
Epoch  120  loss  6.873557826083128 correct 32
Epoch 120 took 2.1620144844055176 seconds
Average time per epoch: 2.100690658427467
Epoch  130  loss  6.402032622742995 correct 31
Epoch 130 took 1.8996670246124268 seconds
Average time per epoch: 2.097687881411487
Epoch  140  loss  6.8641416943887545 correct 31
Epoch 140 took 1.9042744636535645 seconds
Average time per epoch: 2.0973204653313817
Epoch  150  loss  6.86106644082556 correct 31
Epoch 150 took 2.3965485095977783 seconds
Average time per epoch: 2.0947585105895996
Epoch  160  loss  6.208616412152341 correct 31
Epoch 160 took 1.9603931903839111 seconds
Average time per epoch: 2.0957479625014783
Epoch  170  loss  6.763292177798566 correct 31
Epoch 170 took 1.9196243286132812 seconds
Average time per epoch: 2.096711419479192
Epoch  180  loss  6.406851764176262 correct 30
Epoch 180 took 2.563265323638916 seconds
Average time per epoch: 2.0964930347316173
Epoch  190  loss  6.807553258392799 correct 30
Epoch 190 took 1.8955223560333252 seconds
Average time per epoch: 2.0916185466406856
Epoch  200  loss  6.596237909202216 correct 29
Epoch 200 took 1.9569916725158691 seconds
Average time per epoch: 2.0920327006287835
Epoch  210  loss  7.087280670163782 correct 29
Epoch 210 took 2.3476996421813965 seconds
Average time per epoch: 2.091860672873908
Epoch  220  loss  6.2240113456602755 correct 29
Epoch 220 took 1.9287612438201904 seconds
Average time per epoch: 2.087721466478719
Epoch  230  loss  6.537642480259261 correct 31
Epoch 230 took 1.987844705581665 seconds
Average time per epoch: 2.0882610572880997
Epoch  240  loss  6.369458006335815 correct 33
Epoch 240 took 2.047072649002075 seconds
Average time per epoch: 2.0883701935843315
Epoch  250  loss  5.718777374907329 correct 36
Epoch 250 took 1.9111566543579102 seconds
Average time per epoch: 2.085097272082629
Epoch  260  loss  5.551305281385775 correct 36
Epoch 260 took 1.9091441631317139 seconds
Average time per epoch: 2.0858038686701166
Epoch  270  loss  4.945985939666799 correct 36
Epoch 270 took 1.959892749786377 seconds
Average time per epoch: 2.089628655092303
Epoch  280  loss  5.3066428279551925 correct 37
Epoch 280 took 2.315007209777832 seconds
Average time per epoch: 2.08801467987142
Epoch  290  loss  5.523066016783739 correct 40
Epoch 290 took 1.9239931106567383 seconds
Average time per epoch: 2.086919665746263
Epoch  300  loss  4.813189281073752 correct 41
Epoch 300 took 1.921891689300537 seconds
Average time per epoch: 2.0877246341832056
Epoch  310  loss  4.963909122617012 correct 44
Epoch 310 took 2.8060219287872314 seconds
Average time per epoch: 2.0878814884320716
Epoch  320  loss  5.251522477586467 correct 45
Epoch 320 took 1.8877966403961182 seconds
Average time per epoch: 2.085378164814269
Epoch  330  loss  5.905065280900038 correct 45
Epoch 330 took 1.8732435703277588 seconds
Average time per epoch: 2.085010779948393
Epoch  340  loss  4.1042824813168695 correct 44
Epoch 340 took 2.713157892227173 seconds
Average time per epoch: 2.085319382004724
Epoch  350  loss  4.702162561872145 correct 44
Epoch 350 took 1.8962304592132568 seconds
Average time per epoch: 2.082993755313406
Epoch  360  loss  2.488574899276237 correct 45
Epoch 360 took 1.9022266864776611 seconds
Average time per epoch: 2.082907014276182
Epoch  370  loss  3.6858079016210743 correct 46
Epoch 370 took 2.592740535736084 seconds
Average time per epoch: 2.0832117355737405
Epoch  380  loss  3.5824530820742115 correct 46
Epoch 380 took 1.9668524265289307 seconds
Average time per epoch: 2.084007677443697
Epoch  390  loss  4.13820222594114 correct 47
Epoch 390 took 1.9171338081359863 seconds
Average time per epoch: 2.084003469828145
Epoch  400  loss  3.3120272707795135 correct 47
Epoch 400 took 1.8912956714630127 seconds
Average time per epoch: 2.084112573442911
Epoch  410  loss  3.0482871085692005 correct 49
Epoch 410 took 1.8871595859527588 seconds
Average time per epoch: 2.0821731514014177
Epoch  420  loss  2.331552961579362 correct 47
Epoch 420 took 1.9557008743286133 seconds
Average time per epoch: 2.0823435279365956
Epoch  430  loss  3.149307241164894 correct 49
Epoch 430 took 1.936011791229248 seconds
Average time per epoch: 2.082108811823232
Epoch  440  loss  3.013144772926682 correct 47
Epoch 440 took 1.926816701889038 seconds
Average time per epoch: 2.080588786780429
Epoch  450  loss  2.712671260605709 correct 47
Epoch 450 took 2.040593385696411 seconds
Average time per epoch: 2.0810055706294834
Epoch  460  loss  2.0762977321647105 correct 49
Epoch 460 took 1.9440686702728271 seconds
Average time per epoch: 2.0810089550928543
Epoch  470  loss  1.844327482528821 correct 47
Epoch 470 took 2.057828426361084 seconds
Average time per epoch: 2.0800399851140954
Epoch  480  loss  2.3096162924545647 correct 49
Epoch 480 took 1.9035224914550781 seconds
Average time per epoch: 2.0793927781547183
Epoch  490  loss  1.6557371355338957 correct 49
Epoch 490 took 2.0037999153137207 seconds
Average time per epoch: 2.0800141593832104
Average time per epoch: 2.080657030105591
```

### Training on simple dataset

```
Training on simple dataset
Epoch  0  loss  6.747661829344324 correct 25
Epoch 0 took 2.746878147125244 seconds
Average time per epoch: 2.746878147125244
Epoch  10  loss  6.117798417638359 correct 44
Epoch 10 took 1.905862808227539 seconds
Average time per epoch: 2.074728987433694
Epoch  20  loss  5.416899951441451 correct 45
Epoch 20 took 1.9048123359680176 seconds
Average time per epoch: 2.089168514524187
Epoch  30  loss  2.8719513560765355 correct 48
Epoch 30 took 2.8716073036193848 seconds
Average time per epoch: 2.0960904628999772
Epoch  40  loss  2.285557663332948 correct 48
Epoch 40 took 1.8955121040344238 seconds
Average time per epoch: 2.073774558741872
Epoch  50  loss  1.6868096554693905 correct 48
Epoch 50 took 1.8794031143188477 seconds
Average time per epoch: 2.0805504742790673
Epoch  60  loss  1.8670754129398752 correct 48
Epoch 60 took 2.6542110443115234 seconds
Average time per epoch: 2.0835796614162256
Epoch  70  loss  1.6288155447233166 correct 48
Epoch 70 took 1.9670023918151855 seconds
Average time per epoch: 2.073810335615991
Epoch  80  loss  0.4746584034547491 correct 48
Epoch 80 took 1.8896825313568115 seconds
Average time per epoch: 2.076071594968254
Epoch  90  loss  1.7885177147396654 correct 48
Epoch 90 took 2.2982985973358154 seconds
Average time per epoch: 2.0780093250693854
Epoch  100  loss  1.524004248762553 correct 48
Epoch 100 took 1.9882402420043945 seconds
Average time per epoch: 2.0719860449875935
Epoch  110  loss  1.7201853908945044 correct 48
Epoch 110 took 1.919013500213623 seconds
Average time per epoch: 2.0829373179255306
Epoch  120  loss  0.5116254102634209 correct 49
Epoch 120 took 1.884979486465454 seconds
Average time per epoch: 2.0828219524099807
Epoch  130  loss  1.4208565021989148 correct 49
Epoch 130 took 1.987901210784912 seconds
Average time per epoch: 2.076686953770295
Epoch  140  loss  1.554269216974017 correct 48
Epoch 140 took 1.973348617553711 seconds
Average time per epoch: 2.0766980580404297
Epoch  150  loss  0.1870395374876057 correct 49
Epoch 150 took 1.9742021560668945 seconds
Average time per epoch: 2.0779767415381425
Epoch  160  loss  0.049452462912991574 correct 50
Epoch 160 took 2.226017475128174 seconds
Average time per epoch: 2.0753899997805956
Epoch  170  loss  0.28952790144304824 correct 49
Epoch 170 took 1.9184167385101318 seconds
Average time per epoch: 2.0749183272757725
Epoch  180  loss  0.30107532010663396 correct 50
Epoch 180 took 1.9062745571136475 seconds
Average time per epoch: 2.075557724546991
Epoch  190  loss  0.02455186425058504 correct 50
Epoch 190 took 2.5132718086242676 seconds
Average time per epoch: 2.0753384048402
Epoch  200  loss  0.07532900099065942 correct 50
Epoch 200 took 1.8952341079711914 seconds
Average time per epoch: 2.0730677599930645
Epoch  210  loss  0.6902692294114068 correct 50
Epoch 210 took 1.9844846725463867 seconds
Average time per epoch: 2.0738707244113725
Epoch  220  loss  0.2838626357927442 correct 50
Epoch 220 took 2.698955535888672 seconds
Average time per epoch: 2.0737259236935577
Epoch  230  loss  0.24389564837400426 correct 50
Epoch 230 took 1.9492809772491455 seconds
Average time per epoch: 2.0745744818732854
Epoch  240  loss  0.05771476246755007 correct 50
Epoch 240 took 1.8929932117462158 seconds
Average time per epoch: 2.0744520470314503
Epoch  250  loss  0.07638534620742382 correct 50
Epoch 250 took 2.494948387145996 seconds
Average time per epoch: 2.074631352823569
Epoch  260  loss  0.004028369905825104 correct 50
Epoch 260 took 1.9775524139404297 seconds
Average time per epoch: 2.072020315119133
Epoch  270  loss  0.17571085228146305 correct 50
Epoch 270 took 1.8851923942565918 seconds
Average time per epoch: 2.073051891643623
Epoch  280  loss  0.37207907338205465 correct 50
Epoch 280 took 2.139948844909668 seconds
Average time per epoch: 2.0735941247159477
Epoch  290  loss  0.6811295860653019 correct 50
Epoch 290 took 1.9561784267425537 seconds
Average time per epoch: 2.071157630776212
Epoch  300  loss  0.2124797666687467 correct 50
Epoch 300 took 1.8812553882598877 seconds
Average time per epoch: 2.071251675140026
Epoch  310  loss  0.1153876001061678 correct 50
Epoch 310 took 1.9732909202575684 seconds
Average time per epoch: 2.0724404716798346
Epoch  320  loss  0.01974113135637435 correct 50
Epoch 320 took 1.8839058876037598 seconds
Average time per epoch: 2.0701245392594383
Epoch  330  loss  0.38978059826011946 correct 50
Epoch 330 took 1.8663592338562012 seconds
Average time per epoch: 2.0704691309223002
Epoch  340  loss  0.15042640666489915 correct 50
Epoch 340 took 2.7740495204925537 seconds
Average time per epoch: 2.073662107641047
Epoch  350  loss  0.08512112380401066 correct 50
Epoch 350 took 1.8847229480743408 seconds
Average time per epoch: 2.0711805134417323
Epoch  360  loss  0.4139507655480255 correct 50
Epoch 360 took 1.9675841331481934 seconds
Average time per epoch: 2.071763827199751
Epoch  370  loss  0.0659515304665697 correct 50
Epoch 370 took 2.3489692211151123 seconds
Average time per epoch: 2.072162542703017
Epoch  380  loss  0.31433835159976176 correct 50
Epoch 380 took 1.8925390243530273 seconds
Average time per epoch: 2.070466752440285
Epoch  390  loss  0.029013937296620135 correct 50
Epoch 390 took 1.9146687984466553 seconds
Average time per epoch: 2.0708891668588
Epoch  400  loss  0.3099640109251261 correct 50
Epoch 400 took 2.237964391708374 seconds
Average time per epoch: 2.071908216524005
Epoch  410  loss  0.09922773185245186 correct 50
Epoch 410 took 1.8859052658081055 seconds
Average time per epoch: 2.070080845315381
Epoch  420  loss  0.05800155767846638 correct 50
Epoch 420 took 1.9634146690368652 seconds
Average time per epoch: 2.070554814825715
Epoch  430  loss  0.01926262094129405 correct 50
Epoch 430 took 2.0894148349761963 seconds
Average time per epoch: 2.071207225737605
Epoch  440  loss  0.3233693812456864 correct 50
Epoch 440 took 2.035428524017334 seconds
Average time per epoch: 2.070159499066757
Epoch  450  loss  0.01849340757392752 correct 50
Epoch 450 took 1.9269137382507324 seconds
Average time per epoch: 2.070148454801471
Epoch  460  loss  0.06996271270784839 correct 50
Epoch 460 took 1.8773295879364014 seconds
Average time per epoch: 2.072274884536313
Epoch  470  loss  0.03336435381257318 correct 50
Epoch 470 took 2.298426866531372 seconds
Average time per epoch: 2.0713110876184615
Epoch  480  loss  0.05579077871656915 correct 50
Epoch 480 took 1.8783376216888428 seconds
Average time per epoch: 2.0704289687150728
Epoch  490  loss  0.24982700352249892 correct 50
Epoch 490 took 1.878643274307251 seconds
Average time per epoch: 2.0705688966024676
```

### Training on split dataset

```
Training on split dataset
Epoch  0  loss  7.412751894027275 correct 31
Epoch 0 took 0.06581258773803711 seconds
Average time per epoch: 0.06581258773803711
Epoch  10  loss  7.438690967745835 correct 31
Epoch 10 took 0.07817816734313965 seconds
Average time per epoch: 0.06540138071233575
Epoch  20  loss  6.0163111215132385 correct 31
Epoch 20 took 0.07179665565490723 seconds
Average time per epoch: 0.0663611888885498
Epoch  30  loss  6.283312029804888 correct 31
Epoch 30 took 0.06335043907165527 seconds
Average time per epoch: 0.06702669974296324
Epoch  40  loss  9.014860119818211 correct 32
Epoch 40 took 0.07486557960510254 seconds
Average time per epoch: 0.06659095461775617
Epoch  50  loss  4.901001436573224 correct 33
Epoch 50 took 0.06524443626403809 seconds
Average time per epoch: 0.06649990175284591
Epoch  60  loss  5.741190316636601 correct 34
Epoch 60 took 0.06491899490356445 seconds
Average time per epoch: 0.06657412403919657
Epoch  70  loss  7.2890766619119045 correct 37
Epoch 70 took 0.06535792350769043 seconds
Average time per epoch: 0.06633267604129416
Epoch  80  loss  5.365195544272222 correct 41
Epoch 80 took 0.07104849815368652 seconds
Average time per epoch: 0.06650681848879214
Epoch  90  loss  5.196278947162058 correct 42
Epoch 90 took 0.06349563598632812 seconds
Average time per epoch: 0.06669131216112074
Epoch  100  loss  5.9687396504513295 correct 43
Epoch 100 took 0.06742572784423828 seconds
Average time per epoch: 0.06678563061327038
Epoch  110  loss  4.4679100419644495 correct 43
Epoch 110 took 0.06415486335754395 seconds
Average time per epoch: 0.06696714152086962
Epoch  120  loss  4.786574685405981 correct 43
Epoch 120 took 0.06775736808776855 seconds
Average time per epoch: 0.06696392287892744
Epoch  130  loss  4.725509534389245 correct 44
Epoch 130 took 0.06716728210449219 seconds
Average time per epoch: 0.06683947657810822
Epoch  140  loss  7.6888006074310855 correct 44
Epoch 140 took 0.0675961971282959 seconds
Average time per epoch: 0.06683421811313493
Epoch  150  loss  6.346363709972217 correct 44
Epoch 150 took 0.10196089744567871 seconds
Average time per epoch: 0.06736488847543072
Epoch  160  loss  4.436495206414818 correct 44
Epoch 160 took 0.06321501731872559 seconds
Average time per epoch: 0.0691827913248761
Epoch  170  loss  3.229374271470522 correct 44
Epoch 170 took 0.07622098922729492 seconds
Average time per epoch: 0.07040020039207057
Epoch  180  loss  6.954712116339445 correct 44
Epoch 180 took 0.07194757461547852 seconds
Average time per epoch: 0.07236279703635537
Epoch  190  loss  4.487749666575044 correct 44
Epoch 190 took 0.07240557670593262 seconds
Average time per epoch: 0.07201894290784266
Epoch  200  loss  4.439417283365564 correct 44
Epoch 200 took 0.062369346618652344 seconds
Average time per epoch: 0.0716744358859845
Epoch  210  loss  4.087956366128147 correct 44
Epoch 210 took 0.07218050956726074 seconds
Average time per epoch: 0.07139248757565755
Epoch  220  loss  4.3054751403064015 correct 44
Epoch 220 took 0.0673213005065918 seconds
Average time per epoch: 0.07117690867428327
Epoch  230  loss  2.577633172907294 correct 44
Epoch 230 took 0.06513261795043945 seconds
Average time per epoch: 0.07095080020623806
Epoch  240  loss  3.317906117049748 correct 44
Epoch 240 took 0.06435656547546387 seconds
Average time per epoch: 0.07070872892482646
Epoch  250  loss  2.9099341697402963 correct 44
Epoch 250 took 0.06370687484741211 seconds
Average time per epoch: 0.07063658018986067
Epoch  260  loss  4.246121202878502 correct 44
Epoch 260 took 0.07087922096252441 seconds
Average time per epoch: 0.07066681101861128
Epoch  270  loss  1.6514864671822675 correct 44
Epoch 270 took 0.06366610527038574 seconds
Average time per epoch: 0.07048571681624409
Epoch  280  loss  1.8702054378130437 correct 45
Epoch 280 took 0.06467652320861816 seconds
Average time per epoch: 0.07037641993621066
Epoch  290  loss  2.4847005858334645 correct 45
Epoch 290 took 0.06519126892089844 seconds
Average time per epoch: 0.07025479041423995
Epoch  300  loss  1.6404579008277027 correct 46
Epoch 300 took 0.06880664825439453 seconds
Average time per epoch: 0.07011893500521334
Epoch  310  loss  2.2353264749370947 correct 47
Epoch 310 took 0.06662654876708984 seconds
Average time per epoch: 0.07000025292301484
Epoch  320  loss  1.8738338040592801 correct 47
Epoch 320 took 0.06511402130126953 seconds
Average time per epoch: 0.06989695869873618
Epoch  330  loss  1.0309026466314484 correct 47
Epoch 330 took 0.12318873405456543 seconds
Average time per epoch: 0.07033218501917905
Epoch  340  loss  1.0300626336226655 correct 47
Epoch 340 took 0.14020848274230957 seconds
Average time per epoch: 0.07182374448021137
Epoch  350  loss  2.0515370735993814 correct 48
Epoch 350 took 0.14758563041687012 seconds
Average time per epoch: 0.07273428800099256
Epoch  360  loss  1.8336525719240466 correct 49
Epoch 360 took 0.07346582412719727 seconds
Average time per epoch: 0.07296556118782868
Epoch  370  loss  1.8064201687908423 correct 49
Epoch 370 took 0.06406927108764648 seconds
Average time per epoch: 0.07279525677148865
Epoch  380  loss  1.1859234676078279 correct 50
Epoch 380 took 0.06471061706542969 seconds
Average time per epoch: 0.07264210232912399
Epoch  390  loss  1.540938623426096 correct 48
Epoch 390 took 0.06621503829956055 seconds
Average time per epoch: 0.0724987062956671
Epoch  400  loss  1.1873470896051548 correct 50
Epoch 400 took 0.06486177444458008 seconds
Average time per epoch: 0.07239648826104447
Epoch  410  loss  1.0975841430994235 correct 50
Epoch 410 took 0.06430292129516602 seconds
Average time per epoch: 0.07225092310104927
Epoch  420  loss  1.1261894645283097 correct 50
Epoch 420 took 0.06253933906555176 seconds
Average time per epoch: 0.0720559808817159
Epoch  430  loss  1.8934686557156868 correct 48
Epoch 430 took 0.06576991081237793 seconds
Average time per epoch: 0.07194106540104629
Epoch  440  loss  0.6278136612788389 correct 50
Epoch 440 took 0.06613993644714355 seconds
Average time per epoch: 0.0717393751858043
Epoch  450  loss  1.564617183790252 correct 50
Epoch 450 took 0.06385135650634766 seconds
Average time per epoch: 0.07154454677437996
Epoch  460  loss  0.9352030538841899 correct 50
Epoch 460 took 0.06460118293762207 seconds
Average time per epoch: 0.07144032900348919
Epoch  470  loss  1.6409078163735342 correct 50
Epoch 470 took 0.06339836120605469 seconds
Average time per epoch: 0.0712662482211038
Epoch  480  loss  0.5864569299824003 correct 49
Epoch 480 took 0.07052350044250488 seconds
Average time per epoch: 0.07117705583076715
Epoch  490  loss  0.7786298129861697 correct 49
Epoch 490 took 0.06592130661010742 seconds
Average time per epoch: 0.07109696462052417
Average time per epoch: 0.07097361707687377
```

## CPU Training Results

### Training on xor dataset

```
Epoch  0  loss  7.128623081056464 correct 23
Epoch 0 took 21.526084661483765 seconds
Average time per epoch: 21.526084661483765
Epoch  10  loss  6.831736996229117 correct 24
Epoch 10 took 0.06602072715759277 seconds
Average time per epoch: 2.0166721777482466
Epoch  20  loss  6.895936636429019 correct 26
Epoch 20 took 0.06484723091125488 seconds
Average time per epoch: 1.09053981871832
Epoch  30  loss  6.786167461085067 correct 27
Epoch 30 took 0.06355977058410645 seconds
Average time per epoch: 0.7607850336259411
Epoch  40  loss  6.905155874049534 correct 28
Epoch 40 took 0.0716552734375 seconds
Average time per epoch: 0.5907856487646336
Epoch  50  loss  7.016956659066072 correct 28
Epoch 50 took 0.0725715160369873 seconds
Average time per epoch: 0.4885541598002116
Epoch  60  loss  6.695729311759556 correct 29
Epoch 60 took 0.06436467170715332 seconds
Average time per epoch: 0.41951052868952515
Epoch  70  loss  6.71173562703171 correct 30
Epoch 70 took 0.06635022163391113 seconds
Average time per epoch: 0.36964093799322423
Epoch  80  loss  6.513941494987075 correct 30
Epoch 80 took 0.06464886665344238 seconds
Average time per epoch: 0.33219479042806743
Epoch  90  loss  6.463505798844617 correct 30
Epoch 90 took 0.0640726089477539 seconds
Average time per epoch: 0.3029084546225412
Epoch  100  loss  6.618696968085951 correct 31
Epoch 100 took 0.12695550918579102 seconds
Average time per epoch: 0.2823129219583946
Epoch  110  loss  6.029046066462385 correct 30
Epoch 110 took 0.1064450740814209 seconds
Average time per epoch: 0.26558587787387605
Epoch  120  loss  6.071718158609073 correct 31
Epoch 120 took 0.12419486045837402 seconds
Average time per epoch: 0.2534403465996104
Epoch  130  loss  6.852577502341214 correct 32
Epoch 130 took 0.06454277038574219 seconds
Average time per epoch: 0.23903379549506967
Epoch  140  loss  5.796075218888241 correct 32
Epoch 140 took 0.06370186805725098 seconds
Average time per epoch: 0.22683662894769763
Epoch  150  loss  6.594451053010604 correct 33
Epoch 150 took 0.06770801544189453 seconds
Average time per epoch: 0.21622536829765268
Epoch  160  loss  6.081456870706713 correct 35
Epoch 160 took 0.07466483116149902 seconds
Average time per epoch: 0.20700166092155883
Epoch  170  loss  5.5554218194717695 correct 37
Epoch 170 took 0.06387448310852051 seconds
Average time per epoch: 0.19878857595878735
Epoch  180  loss  5.224201528480103 correct 41
Epoch 180 took 0.0664372444152832 seconds
Average time per epoch: 0.19143445715719823
Epoch  190  loss  5.081466332203561 correct 43
Epoch 190 took 0.06550335884094238 seconds
Average time per epoch: 0.1848565421179327
Epoch  200  loss  5.043573723039724 correct 43
Epoch 200 took 0.06245255470275879 seconds
Average time per epoch: 0.17896475602145218
Epoch  210  loss  4.831146591312516 correct 45
Epoch 210 took 0.06429696083068848 seconds
Average time per epoch: 0.17348134009194036
Epoch  220  loss  4.315427254182679 correct 46
Epoch 220 took 0.06355452537536621 seconds
Average time per epoch: 0.16863243288583885
Epoch  230  loss  4.310210507527965 correct 45
Epoch 230 took 0.07211589813232422 seconds
Average time per epoch: 0.16435863754966043
Epoch  240  loss  3.8411907859341508 correct 45
Epoch 240 took 0.06369185447692871 seconds
Average time per epoch: 0.1602490838632544
Epoch  250  loss  3.4377849939131666 correct 44
Epoch 250 took 0.06456542015075684 seconds
Average time per epoch: 0.15651032839163367
Epoch  260  loss  4.860316244967121 correct 44
Epoch 260 took 0.07337117195129395 seconds
Average time per epoch: 0.15309647125302603
Epoch  270  loss  4.829306373237226 correct 46
Epoch 270 took 0.07743024826049805 seconds
Average time per epoch: 0.15006095984765083
Epoch  280  loss  4.121290743611578 correct 46
Epoch 280 took 0.06254100799560547 seconds
Average time per epoch: 0.14806656905340554
Epoch  290  loss  3.4615573234376593 correct 46
Epoch 290 took 0.13537955284118652 seconds
Average time per epoch: 0.14641532701315338
Epoch  300  loss  3.697803713688869 correct 46
Epoch 300 took 0.06462979316711426 seconds
Average time per epoch: 0.1452456708762337
Epoch  310  loss  3.195919666883735 correct 45
Epoch 310 took 0.06557202339172363 seconds
Average time per epoch: 0.14274535071811492
Epoch  320  loss  2.268817079786648 correct 46
Epoch 320 took 0.06193733215332031 seconds
Average time per epoch: 0.14033844834918915
Epoch  330  loss  3.0086202787693597 correct 48
Epoch 330 took 0.06528282165527344 seconds
Average time per epoch: 0.13801396361290508
Epoch  340  loss  2.7075977776896236 correct 47
Epoch 340 took 0.06388568878173828 seconds
Average time per epoch: 0.13589202483728136
Epoch  350  loss  1.7308687299163272 correct 49
Epoch 350 took 0.0634012222290039 seconds
Average time per epoch: 0.13383643239991277
Epoch  360  loss  1.2619110236283397 correct 47
Epoch 360 took 0.0650336742401123 seconds
Average time per epoch: 0.1320389604964745
Epoch  370  loss  1.0440526809685144 correct 49
Epoch 370 took 0.06504440307617188 seconds
Average time per epoch: 0.1303291931306577
Epoch  380  loss  1.6529040729439344 correct 48
Epoch 380 took 0.06601786613464355 seconds
Average time per epoch: 0.12862036171860583
Epoch  390  loss  2.4952460738330564 correct 49
Epoch 390 took 0.06511044502258301 seconds
Average time per epoch: 0.12707595691046752
Epoch  400  loss  1.3199864004097095 correct 49
Epoch 400 took 0.06685376167297363 seconds
Average time per epoch: 0.12556665258811894
Epoch  410  loss  1.6405399230812938 correct 48
Epoch 410 took 0.0631096363067627 seconds
Average time per epoch: 0.12413557660550677
Epoch  420  loss  0.7948635489854 correct 49
Epoch 420 took 0.06557130813598633 seconds
Average time per epoch: 0.12274880590461495
Epoch  430  loss  0.9687336229730726 correct 47
Epoch 430 took 0.06559443473815918 seconds
Average time per epoch: 0.12146913590398023
Epoch  440  loss  2.1608696204894082 correct 46
Epoch 440 took 0.06258201599121094 seconds
Average time per epoch: 0.12017329533894856
Epoch  450  loss  4.843925916698734 correct 43
Epoch 450 took 0.09679055213928223 seconds
Average time per epoch: 0.1191823218191278
Epoch  460  loss  1.475627641932939 correct 50
Epoch 460 took 0.12004899978637695 seconds
Average time per epoch: 0.11918448005477673
Epoch  470  loss  0.7170769706749399 correct 49
Epoch 470 took 0.13546323776245117 seconds
Average time per epoch: 0.1188675061406097
Epoch  480  loss  0.4321616444256393 correct 50
Epoch 480 took 0.0674736499786377 seconds
Average time per epoch: 0.11819542321742449
Epoch  490  loss  1.196481902552034 correct 49
Epoch 490 took 0.06322741508483887 seconds
Average time per epoch: 0.1171310598641452
Average time per epoch: 0.11617024898529053
```

### Training on simple dataset

```
Epoch  0  loss  6.770171001590719 correct 35
Epoch 0 took 0.0634465217590332 seconds
Average time per epoch: 0.0634465217590332
Epoch  10  loss  6.191395332358581 correct 47
Epoch 10 took 0.06341433525085449 seconds
Average time per epoch: 0.06741313500837846
Epoch  20  loss  5.855352829603716 correct 47
Epoch 20 took 0.07005000114440918 seconds
Average time per epoch: 0.06763037045796712
Epoch  30  loss  5.008712188035883 correct 49
Epoch 30 took 0.07383275032043457 seconds
Average time per epoch: 0.06782970889922112
Epoch  40  loss  3.401400673962043 correct 49
Epoch 40 took 0.06255936622619629 seconds
Average time per epoch: 0.0676384902581936
Epoch  50  loss  2.858829538892344 correct 50
Epoch 50 took 0.07275271415710449 seconds
Average time per epoch: 0.06760255963194604
Epoch  60  loss  1.1915455623130826 correct 49
Epoch 60 took 0.06488680839538574 seconds
Average time per epoch: 0.06750805651555296
Epoch  70  loss  1.3179480936468573 correct 50
Epoch 70 took 0.06287145614624023 seconds
Average time per epoch: 0.0673391986900652
Epoch  80  loss  2.512616457886609 correct 49
Epoch 80 took 0.06459593772888184 seconds
Average time per epoch: 0.06729737328894345
Epoch  90  loss  1.3859078086536358 correct 50
Epoch 90 took 0.06815862655639648 seconds
Average time per epoch: 0.06747756947527875
Epoch  100  loss  0.7954107672074763 correct 50
Epoch 100 took 0.0675971508026123 seconds
Average time per epoch: 0.06749407607730072
Epoch  110  loss  0.9839648568209022 correct 50
Epoch 110 took 0.06386899948120117 seconds
Average time per epoch: 0.06736548526867016
Epoch  120  loss  1.405342606097686 correct 50
Epoch 120 took 0.06630158424377441 seconds
Average time per epoch: 0.06759923154657538
Epoch  130  loss  1.0329364664308285 correct 50
Epoch 130 took 0.10244250297546387 seconds
Average time per epoch: 0.06856179237365723
Epoch  140  loss  1.386438879115042 correct 50
Epoch 140 took 0.14008450508117676 seconds
Average time per epoch: 0.07094968295266442
Epoch  150  loss  0.8004439702675933 correct 50
Epoch 150 took 0.1431586742401123 seconds
Average time per epoch: 0.07327362401595969
Epoch  160  loss  0.6012074795017852 correct 50
Epoch 160 took 0.07146406173706055 seconds
Average time per epoch: 0.07357619119727093
Epoch  170  loss  0.2191272064490253 correct 50
Epoch 170 took 0.06593084335327148 seconds
Average time per epoch: 0.07311608080278363
Epoch  180  loss  0.30142882720926534 correct 50
Epoch 180 took 0.06770753860473633 seconds
Average time per epoch: 0.07278021001025459
Epoch  190  loss  0.12778826466090665 correct 50
Epoch 190 took 0.07246994972229004 seconds
Average time per epoch: 0.07254059027626876
Epoch  200  loss  0.1301241818904892 correct 50
Epoch 200 took 0.06514525413513184 seconds
Average time per epoch: 0.07221282536710673
Epoch  210  loss  0.7713642256180603 correct 50
Epoch 210 took 0.06374335289001465 seconds
Average time per epoch: 0.07189210670254242
Epoch  220  loss  0.19431981582793204 correct 50
Epoch 220 took 0.06741905212402344 seconds
Average time per epoch: 0.07162576886863191
Epoch  230  loss  0.773283460182525 correct 50
Epoch 230 took 0.06490540504455566 seconds
Average time per epoch: 0.0713629320070341
Epoch  240  loss  0.22540820728293742 correct 50
Epoch 240 took 0.07115936279296875 seconds
Average time per epoch: 0.0713494922115595
Epoch  250  loss  0.014899322313738342 correct 50
Epoch 250 took 0.06418919563293457 seconds
Average time per epoch: 0.07111659752895158
Epoch  260  loss  0.4539558280164111 correct 50
Epoch 260 took 0.0648348331451416 seconds
Average time per epoch: 0.07090989200548194
Epoch  270  loss  0.08038939655565769 correct 50
Epoch 270 took 0.07009124755859375 seconds
Average time per epoch: 0.07074670510098503
Epoch  280  loss  0.10070768501574352 correct 50
Epoch 280 took 0.06355667114257812 seconds
Average time per epoch: 0.07053244326038292
Epoch  290  loss  0.10890454047503739 correct 50
Epoch 290 took 0.06622123718261719 seconds
Average time per epoch: 0.07044911876167219
Epoch  300  loss  0.8542712147845873 correct 50
Epoch 300 took 0.06426024436950684 seconds
Average time per epoch: 0.07034479502427618
Epoch  310  loss  0.1449011009752129 correct 50
Epoch 310 took 0.11711931228637695 seconds
Average time per epoch: 0.07155655357998666
Epoch  320  loss  0.007479535630428617 correct 50
Epoch 320 took 0.09545493125915527 seconds
Average time per epoch: 0.0725910968126909
Epoch  330  loss  0.0687150658707598 correct 50
Epoch 330 took 0.06351590156555176 seconds
Average time per epoch: 0.0734793109836175
Epoch  340  loss  0.06173124156632255 correct 50
Epoch 340 took 0.06419038772583008 seconds
Average time per epoch: 0.07326540079983798
Epoch  350  loss  0.41283437881741897 correct 50
Epoch 350 took 0.06676244735717773 seconds
Average time per epoch: 0.07312967906310687
Epoch  360  loss  0.6251210561576415 correct 50
Epoch 360 took 0.06440544128417969 seconds
Average time per epoch: 0.07292055357195994
Epoch  370  loss  0.008983378300419674 correct 50
Epoch 370 took 0.0635528564453125 seconds
Average time per epoch: 0.07305386509856766
Epoch  380  loss  0.6378583547279699 correct 50
Epoch 380 took 0.06429147720336914 seconds
Average time per epoch: 0.07291564791221318
Epoch  390  loss  0.0226777331195036 correct 50
Epoch 390 took 0.06314897537231445 seconds
Average time per epoch: 0.07274774517244695
Epoch  400  loss  0.8580359351467896 correct 50
Epoch 400 took 0.0635836124420166 seconds
Average time per epoch: 0.07257443651594128
Epoch  410  loss  0.08708173918859878 correct 50
Epoch 410 took 0.06624817848205566 seconds
Average time per epoch: 0.07243540803301364
Epoch  420  loss  0.03746370670022216 correct 50
Epoch 420 took 0.06851983070373535 seconds
Average time per epoch: 0.0723229349367409
Epoch  430  loss  0.4965232947836344 correct 50
Epoch 430 took 0.0644674301147461 seconds
Average time per epoch: 0.0722152948932692
Epoch  440  loss  0.008074241719473432 correct 50
Epoch 440 took 0.07280874252319336 seconds
Average time per epoch: 0.07209758585542779
Epoch  450  loss  0.8135405624794132 correct 50
Epoch 450 took 0.06431055068969727 seconds
Average time per epoch: 0.07201573219637648
Epoch  460  loss  0.6177585262543324 correct 50
Epoch 460 took 0.06425857543945312 seconds
Average time per epoch: 0.07184247287907466
Epoch  470  loss  0.21494535336011605 correct 50
Epoch 470 took 0.06512188911437988 seconds
Average time per epoch: 0.0717546529830641
Epoch  480  loss  0.07336323619522872 correct 50
Epoch 480 took 0.06860828399658203 seconds
Average time per epoch: 0.07200799662457187
Epoch  490  loss  0.04003529983389733 correct 50
Epoch 490 took 0.08365654945373535 seconds
Average time per epoch: 0.07301551989770955
Average time per epoch: 0.0738464560508728
```

### Training on split dataset

```
Epoch  0  loss  7.412751894027275 correct 31
Epoch 0 took 0.06581258773803711 seconds
Average time per epoch: 0.06581258773803711
Epoch  10  loss  7.438690967745835 correct 31
Epoch 10 took 0.07817816734313965 seconds
Average time per epoch: 0.06540138071233575
Epoch  20  loss  6.0163111215132385 correct 31
Epoch 20 took 0.07179665565490723 seconds
Average time per epoch: 0.0663611888885498
Epoch  30  loss  6.283312029804888 correct 31
Epoch 30 took 0.06335043907165527 seconds
Average time per epoch: 0.06702669974296324
Epoch  40  loss  9.014860119818211 correct 32
Epoch 40 took 0.07486557960510254 seconds
Average time per epoch: 0.06659095461775617
Epoch  50  loss  4.901001436573224 correct 33
Epoch 50 took 0.06524443626403809 seconds
Average time per epoch: 0.06649990175284591
Epoch  60  loss  5.741190316636601 correct 34
Epoch 60 took 0.06491899490356445 seconds
Average time per epoch: 0.06657412403919657
Epoch  70  loss  7.2890766619119045 correct 37
Epoch 70 took 0.06535792350769043 seconds
Average time per epoch: 0.06633267604129416
Epoch  80  loss  5.365195544272222 correct 41
Epoch 80 took 0.07104849815368652 seconds
Average time per epoch: 0.06650681848879214
Epoch  90  loss  5.196278947162058 correct 42
Epoch 90 took 0.06349563598632812 seconds
Average time per epoch: 0.06669131216112074
Epoch  100  loss  5.9687396504513295 correct 43
Epoch 100 took 0.06742572784423828 seconds
Average time per epoch: 0.06678563061327038
Epoch  110  loss  4.4679100419644495 correct 43
Epoch 110 took 0.06415486335754395 seconds
Average time per epoch: 0.06696714152086962
Epoch  120  loss  4.786574685405981 correct 43
Epoch 120 took 0.06775736808776855 seconds
Average time per epoch: 0.06696392287892744
Epoch  130  loss  4.725509534389245 correct 44
Epoch 130 took 0.06716728210449219 seconds
Average time per epoch: 0.06683947657810822
Epoch  140  loss  7.6888006074310855 correct 44
Epoch 140 took 0.0675961971282959 seconds
Average time per epoch: 0.06683421811313493
Epoch  150  loss  6.346363709972217 correct 44
Epoch 150 took 0.10196089744567871 seconds
Average time per epoch: 0.06736488847543072
Epoch  160  loss  4.436495206414818 correct 44
Epoch 160 took 0.06321501731872559 seconds
Average time per epoch: 0.0691827913248761
Epoch  170  loss  3.229374271470522 correct 44
Epoch 170 took 0.07622098922729492 seconds
Average time per epoch: 0.07040020039207057
Epoch  180  loss  6.954712116339445 correct 44
Epoch 180 took 0.07194757461547852 seconds
Average time per epoch: 0.07236279703635537
Epoch  190  loss  4.487749666575044 correct 44
Epoch 190 took 0.07240557670593262 seconds
Average time per epoch: 0.07201894290784266
Epoch  200  loss  4.439417283365564 correct 44
Epoch 200 took 0.062369346618652344 seconds
Average time per epoch: 0.0716744358859845
Epoch  210  loss  4.087956366128147 correct 44
Epoch 210 took 0.07218050956726074 seconds
Average time per epoch: 0.07139248757565755
Epoch  220  loss  4.3054751403064015 correct 44
Epoch 220 took 0.0673213005065918 seconds
Average time per epoch: 0.07117690867428327
Epoch  230  loss  2.577633172907294 correct 44
Epoch 230 took 0.06513261795043945 seconds
Average time per epoch: 0.07095080020623806
Epoch  240  loss  3.317906117049748 correct 44
Epoch 240 took 0.06435656547546387 seconds
Average time per epoch: 0.07070872892482646
Epoch  250  loss  2.9099341697402963 correct 44
Epoch 250 took 0.06370687484741211 seconds
Average time per epoch: 0.07063658018986067
Epoch  260  loss  4.246121202878502 correct 44
Epoch 260 took 0.07087922096252441 seconds
Average time per epoch: 0.07066681101861128
Epoch  270  loss  1.6514864671822675 correct 44
Epoch 270 took 0.06366610527038574 seconds
Average time per epoch: 0.07048571681624409
Epoch  280  loss  1.8702054378130437 correct 45
Epoch 280 took 0.06467652320861816 seconds
Average time per epoch: 0.07037641993621066
Epoch  290  loss  2.4847005858334645 correct 45
Epoch 290 took 0.06519126892089844 seconds
Average time per epoch: 0.07025479041423995
Epoch  300  loss  1.6404579008277027 correct 46
Epoch 300 took 0.06880664825439453 seconds
Average time per epoch: 0.07011893500521334
Epoch  310  loss  2.2353264749370947 correct 47
Epoch 310 took 0.06662654876708984 seconds
Average time per epoch: 0.07000025292301484
Epoch  320  loss  1.8738338040592801 correct 47
Epoch 320 took 0.06511402130126953 seconds
Average time per epoch: 0.06989695869873618
Epoch  330  loss  1.0309026466314484 correct 47
Epoch 330 took 0.12318873405456543 seconds
Average time per epoch: 0.07033218501917905
Epoch  340  loss  1.0300626336226655 correct 47
Epoch 340 took 0.14020848274230957 seconds
Average time per epoch: 0.07182374448021137
Epoch  350  loss  2.0515370735993814 correct 48
Epoch 350 took 0.14758563041687012 seconds
Average time per epoch: 0.07273428800099256
Epoch  360  loss  1.8336525719240466 correct 49
Epoch 360 took 0.07346582412719727 seconds
Average time per epoch: 0.07296556118782868
Epoch  370  loss  1.8064201687908423 correct 49
Epoch 370 took 0.06406927108764648 seconds
Average time per epoch: 0.07279525677148865
Epoch  380  loss  1.1859234676078279 correct 50
Epoch 380 took 0.06471061706542969 seconds
Average time per epoch: 0.07264210232912399
Epoch  390  loss  1.540938623426096 correct 48
Epoch 390 took 0.06621503829956055 seconds
Average time per epoch: 0.0724987062956671
Epoch  400  loss  1.1873470896051548 correct 50
Epoch 400 took 0.06486177444458008 seconds
Average time per epoch: 0.07239648826104447
Epoch  410  loss  1.0975841430994235 correct 50
Epoch 410 took 0.06430292129516602 seconds
Average time per epoch: 0.07225092310104927
Epoch  420  loss  1.1261894645283097 correct 50
Epoch 420 took 0.06253933906555176 seconds
Average time per epoch: 0.0720559808817159
Epoch  430  loss  1.8934686557156868 correct 48
Epoch 430 took 0.06576991081237793 seconds
Average time per epoch: 0.07194106540104629
Epoch  440  loss  0.6278136612788389 correct 50
Epoch 440 took 0.06613993644714355 seconds
Average time per epoch: 0.0717393751858043
Epoch  450  loss  1.564617183790252 correct 50
Epoch 450 took 0.06385135650634766 seconds
Average time per epoch: 0.07154454677437996
Epoch  460  loss  0.9352030538841899 correct 50
Epoch 460 took 0.06460118293762207 seconds
Average time per epoch: 0.07144032900348919
Epoch  470  loss  1.6409078163735342 correct 50
Epoch 470 took 0.06339836120605469 seconds
Average time per epoch: 0.0712662482211038
Epoch  480  loss  0.5864569299824003 correct 49
Epoch 480 took 0.07052350044250488 seconds
Average time per epoch: 0.07117705583076715
Epoch  490  loss  0.7786298129861697 correct 49
Epoch 490 took 0.06592130661010742 seconds
Average time per epoch: 0.07109696462052417
Average time per epoch: 0.07097361707687377
```

# Large Model Training Results

## CPU 200 Hidden Layers, XOR Dataset

```
Epoch  0  loss  11.916224808889751 correct 32
Epoch 0 took 23.962137699127197 seconds
Average time per epoch: 23.962137699127197
Epoch  10  loss  1.6377792949956147 correct 45
Epoch 10 took 0.3255789279937744 seconds
Average time per epoch: 2.483805634758689
Epoch  20  loss  2.400424100884914 correct 42
Epoch 20 took 0.33786725997924805 seconds
Average time per epoch: 1.5300010272434779
Epoch  30  loss  1.4622929228048531 correct 46
Epoch 30 took 0.3404960632324219 seconds
Average time per epoch: 1.1441645006979666
Epoch  40  loss  2.4028813368937545 correct 46
Epoch 40 took 0.3382725715637207 seconds
Average time per epoch: 0.9461682075407447
Epoch  50  loss  1.6716759453207621 correct 47
Epoch 50 took 0.342684268951416 seconds
Average time per epoch: 0.8542129058463901
Epoch  60  loss  0.7732383107424634 correct 47
Epoch 60 took 0.3333725929260254 seconds
Average time per epoch: 0.7691160069137323
Epoch  70  loss  2.1985167272733017 correct 47
Epoch 70 took 0.3274199962615967 seconds
Average time per epoch: 0.7076612828483044
Epoch  80  loss  1.1841234289808105 correct 48
Epoch 80 took 0.6782248020172119 seconds
Average time per epoch: 0.6668823972160434
Epoch  90  loss  0.7465702921013785 correct 48
Epoch 90 took 0.3304727077484131 seconds
Average time per epoch: 0.641906486762749
Epoch  100  loss  1.4494224201332524 correct 48
Epoch 100 took 0.3757760524749756 seconds
Average time per epoch: 0.6143512891070677
Epoch  110  loss  2.1942369975981006 correct 49
Epoch 110 took 0.35240769386291504 seconds
Average time per epoch: 0.5936207792780421
Epoch  120  loss  1.6272262528238293 correct 48
Epoch 120 took 0.341571569442749 seconds
Average time per epoch: 0.5856513484450411
Epoch  130  loss  1.0994893273383664 correct 48
Epoch 130 took 0.3310105800628662 seconds
Average time per epoch: 0.5665924003106038
Epoch  140  loss  1.909870213985728 correct 49
Epoch 140 took 0.33687663078308105 seconds
Average time per epoch: 0.5503901437664709
Epoch  150  loss  0.9012601432992746 correct 48
Epoch 150 took 0.32600951194763184 seconds
Average time per epoch: 0.5454384001675031
Epoch  160  loss  1.6771553267587351 correct 47
Epoch 160 took 0.3366711139678955 seconds
Average time per epoch: 0.5326383321181588
Epoch  170  loss  0.5211362846547706 correct 49
Epoch 170 took 0.3432748317718506 seconds
Average time per epoch: 0.5211222325152124
Epoch  180  loss  0.6487345092360414 correct 48
Epoch 180 took 0.7171571254730225 seconds
Average time per epoch: 0.5142728887210235
Epoch  190  loss  0.2117443425039515 correct 50
Epoch 190 took 0.3347814083099365 seconds
Average time per epoch: 0.5087834415635514
Epoch  200  loss  1.1697229113519296 correct 48
Epoch 200 took 0.3224663734436035 seconds
Average time per epoch: 0.5000087669239709
Epoch  210  loss  1.396511042728304 correct 48
Epoch 210 took 0.33119654655456543 seconds
Average time per epoch: 0.49203158780861805
Epoch  220  loss  1.0472961112367791 correct 50
Epoch 220 took 0.3344547748565674 seconds
Average time per epoch: 0.4911946678593148
Epoch  230  loss  0.7767018161884922 correct 49
Epoch 230 took 0.3255605697631836 seconds
Average time per epoch: 0.48430633235287357
Epoch  240  loss  0.5696899427419596 correct 49
Epoch 240 took 0.32564234733581543 seconds
Average time per epoch: 0.47816190877890685
Epoch  250  loss  0.8718283508158252 correct 50
Epoch 250 took 0.6539928913116455 seconds
Average time per epoch: 0.4768466740490431
Epoch  260  loss  0.49315808202252726 correct 49
Epoch 260 took 0.3287839889526367 seconds
Average time per epoch: 0.472535944533074
Epoch  270  loss  0.8070054894654886 correct 50
Epoch 270 took 0.3303205966949463 seconds
Average time per epoch: 0.46748154453685803
Epoch  280  loss  0.1985237595737283 correct 50
Epoch 280 took 0.3263280391693115 seconds
Average time per epoch: 0.4627540255780746
Epoch  290  loss  0.8926658143869712 correct 50
Epoch 290 took 0.32917284965515137 seconds
Average time per epoch: 0.4630019582833621
Epoch  300  loss  0.45327650808827186 correct 50
Epoch 300 took 0.31780362129211426 seconds
Average time per epoch: 0.458712600790385
Epoch  310  loss  0.5390741692915724 correct 50
Epoch 310 took 0.3493459224700928 seconds
Average time per epoch: 0.4547753671357869
Epoch  320  loss  1.2993494879701495 correct 48
Epoch 320 took 0.3530154228210449 seconds
Average time per epoch: 0.4555125971820867
Epoch  330  loss  0.14633704279586648 correct 50
Epoch 330 took 0.3252596855163574 seconds
Average time per epoch: 0.4517339802940807
Epoch  340  loss  0.432240844741623 correct 50
Epoch 340 took 0.33907651901245117 seconds
Average time per epoch: 0.4483267652673805
Epoch  350  loss  0.160789930643164 correct 50
Epoch 350 took 0.6433141231536865 seconds
Average time per epoch: 0.44603982270613013
Epoch  360  loss  0.4795179662798305 correct 50
Epoch 360 took 0.3220231533050537 seconds
Average time per epoch: 0.4457462692524918
Epoch  370  loss  1.0207185781323938 correct 50
Epoch 370 took 0.3473963737487793 seconds
Average time per epoch: 0.4426950508693479
Epoch  380  loss  0.43065160132412317 correct 50
Epoch 380 took 0.3247952461242676 seconds
Average time per epoch: 0.4398181644950326
Epoch  390  loss  0.09833145832303206 correct 50
Epoch 390 took 0.3333883285522461 seconds
Average time per epoch: 0.4405871053485919
Epoch  400  loss  1.152902509810902 correct 49
Epoch 400 took 0.3417809009552002 seconds
Average time per epoch: 0.4380406388023548
Epoch  410  loss  0.8214325720192133 correct 50
Epoch 410 took 0.32533955574035645 seconds
Average time per epoch: 0.4354809966400592
Epoch  420  loss  0.9005714506414 correct 50
Epoch 420 took 0.698836088180542 seconds
Average time per epoch: 0.43503623960137083
Epoch  430  loss  0.8597304474951756 correct 50
Epoch 430 took 0.3398091793060303 seconds
Average time per epoch: 0.4341137564099305
Epoch  440  loss  0.07697032423962957 correct 50
Epoch 440 took 0.3254702091217041 seconds
Average time per epoch: 0.4318573691136713
Epoch  450  loss  0.541644929849305 correct 50
Epoch 450 took 0.32807493209838867 seconds
Average time per epoch: 0.42967130815374877
Epoch  460  loss  0.07991176159854084 correct 50
Epoch 460 took 0.3343358039855957 seconds
Average time per epoch: 0.43070695560562894
Epoch  470  loss  0.17494554708580878 correct 50
Epoch 470 took 0.3307814598083496 seconds
Average time per epoch: 0.42870418989987386
Epoch  480  loss  1.1671484078703136 correct 48
Epoch 480 took 0.33115577697753906 seconds
Average time per epoch: 0.4267974995278023
Epoch  490  loss  0.5549378554816999 correct 50
Epoch 490 took 0.6149942874908447 seconds
Average time per epoch: 0.4278138587042659
Average time per epoch: 0.42616982507705686
```

## GPU 200 Hidden Layers, XOR Dataset

```
Epoch  0  loss  14.816589797254391 correct 28
Epoch 0 took 5.243637561798096 seconds
Average time per epoch: 5.243637561798096
Epoch  10  loss  3.42504384340233 correct 44
Epoch 10 took 2.3987820148468018 seconds
Average time per epoch: 2.4535440531643955
Epoch  20  loss  4.024260795564917 correct 43
Epoch 20 took 2.033377170562744 seconds
Average time per epoch: 2.333614076886858
Epoch  30  loss  3.335183605613485 correct 45
Epoch 30 took 2.009803056716919 seconds
Average time per epoch: 2.2936289771910636
Epoch  40  loss  2.63029020642531 correct 45
Epoch 40 took 2.0514156818389893 seconds
Average time per epoch: 2.2746961814601248
Epoch  50  loss  1.895681317943453 correct 46
Epoch 50 took 3.3237977027893066 seconds
Average time per epoch: 2.284237931756412
Epoch  60  loss  3.6632010784446924 correct 47
Epoch 60 took 2.134856700897217 seconds
Average time per epoch: 2.259959979135482
Epoch  70  loss  1.3407686557096743 correct 46
Epoch 70 took 1.9944562911987305 seconds
Average time per epoch: 2.2510888945888468
Epoch  80  loss  1.6307956404687889 correct 47
Epoch 80 took 2.0062360763549805 seconds
Average time per epoch: 2.2440512121459584
Epoch  90  loss  1.6592414358309413 correct 47
Epoch 90 took 2.1393535137176514 seconds
Average time per epoch: 2.241054110474639
Epoch  100  loss  3.100774582552564 correct 47
Epoch 100 took 2.8445098400115967 seconds
Average time per epoch: 2.235746539465272
Epoch  110  loss  1.8437708383240563 correct 48
Epoch 110 took 2.015273094177246 seconds
Average time per epoch: 2.2248376790467685
Epoch  120  loss  1.8856474309975724 correct 47
Epoch 120 took 1.9960594177246094 seconds
Average time per epoch: 2.2219451971290525
Epoch  130  loss  2.1559654780185236 correct 49
Epoch 130 took 2.0098793506622314 seconds
Average time per epoch: 2.2210227059953995
Epoch  140  loss  1.5749224579185084 correct 49
Epoch 140 took 2.560502529144287 seconds
Average time per epoch: 2.219144702803159
Epoch  150  loss  1.8421131410791924 correct 50
Epoch 150 took 2.04215145111084 seconds
Average time per epoch: 2.2118037552233565
Epoch  160  loss  0.6342136822480583 correct 49
Epoch 160 took 2.1524813175201416 seconds
Average time per epoch: 2.21327395172593
Epoch  170  loss  1.1905692946682287 correct 48
Epoch 170 took 2.0443413257598877 seconds
Average time per epoch: 2.2193953600543286
Epoch  180  loss  0.49320362413735147 correct 50
Epoch 180 took 2.0202505588531494 seconds
Average time per epoch: 2.2193455709278256
Epoch  190  loss  0.7772224200825868 correct 50
Epoch 190 took 2.7946202754974365 seconds
Average time per epoch: 2.2190850929440002
Epoch  200  loss  1.6779646939292654 correct 50
Epoch 200 took 2.101224422454834 seconds
Average time per epoch: 2.215602590076959
Epoch  210  loss  1.1487057670745946 correct 48
Epoch 210 took 2.005661964416504 seconds
Average time per epoch: 2.215103500827229
Epoch  220  loss  1.2525810043679764 correct 50
Epoch 220 took 2.0064382553100586 seconds
Average time per epoch: 2.2145135855782625
Epoch  230  loss  0.3800808883766741 correct 50
Epoch 230 took 2.426727533340454 seconds
Average time per epoch: 2.21563619349426
Epoch  240  loss  0.7414043424047666 correct 50
Epoch 240 took 2.4005448818206787 seconds
Average time per epoch: 2.2137652630627898
Epoch  250  loss  1.5704737142181258 correct 50
Epoch 250 took 2.0676567554473877 seconds
Average time per epoch: 2.2123528068284113
Epoch  260  loss  0.06276219788173742 correct 49
Epoch 260 took 2.042919397354126 seconds
Average time per epoch: 2.212737200360645
Epoch  270  loss  1.3211461908293323 correct 50
Epoch 270 took 2.101243734359741 seconds
Average time per epoch: 2.2138002921734348
Epoch  280  loss  0.6743521380497093 correct 50
Epoch 280 took 2.044112205505371 seconds
Average time per epoch: 2.2171019744194274
Epoch  290  loss  1.044458058618178 correct 49
Epoch 290 took 2.5756733417510986 seconds
Average time per epoch: 2.215768010345931
Epoch  300  loss  0.4203258345568883 correct 50
Epoch 300 took 2.030857563018799 seconds
Average time per epoch: 2.2136272251407965
Epoch  310  loss  0.39542084673394023 correct 50
Epoch 310 took 2.0576515197753906 seconds
Average time per epoch: 2.213496778555622
Epoch  320  loss  0.49768993851729226 correct 50
Epoch 320 took 2.013702392578125 seconds
Average time per epoch: 2.2128576608461756
Epoch  330  loss  0.29341677732252636 correct 50
Epoch 330 took 2.632488489151001 seconds
Average time per epoch: 2.212041759058788
Epoch  340  loss  0.13430275434409353 correct 50
Epoch 340 took 2.0431911945343018 seconds
Average time per epoch: 2.208836749152354
Epoch  350  loss  0.1594015299405805 correct 50
Epoch 350 took 1.992300033569336 seconds
Average time per epoch: 2.2084255972479143
Epoch  360  loss  0.45090184152037527 correct 50
Epoch 360 took 1.9976787567138672 seconds
Average time per epoch: 2.206946218443049
Epoch  370  loss  0.12270715844236414 correct 50
Epoch 370 took 2.378230333328247 seconds
Average time per epoch: 2.2054940767365325
Epoch  380  loss  0.47400959648821134 correct 50
Epoch 380 took 2.326511859893799 seconds
Average time per epoch: 2.203670910962923
Epoch  390  loss  0.40317922385215166 correct 50
Epoch 390 took 2.8324644565582275 seconds
Average time per epoch: 2.2044678270969245
Epoch  400  loss  0.47801873623606284 correct 50
Epoch 400 took 1.9935343265533447 seconds
Average time per epoch: 2.2036443553363294
Epoch  410  loss  0.4338988065206888 correct 50
Epoch 410 took 1.9987311363220215 seconds
Average time per epoch: 2.202382554683082
Epoch  420  loss  0.5073066967930786 correct 50
Epoch 420 took 2.6040425300598145 seconds
Average time per epoch: 2.2007516653690655
Epoch  430  loss  0.8345573254203196 correct 50
Epoch 430 took 1.9662847518920898 seconds
Average time per epoch: 2.1979912986888133
Epoch  440  loss  0.42314560526790934 correct 50
Epoch 440 took 1.9704067707061768 seconds
Average time per epoch: 2.197111986121353
Epoch  450  loss  0.07529950352898522 correct 50
Epoch 450 took 2.022592544555664 seconds
Average time per epoch: 2.1965251830623314
Epoch  460  loss  0.42843502884856816 correct 50
Epoch 460 took 2.5502052307128906 seconds
Average time per epoch: 2.195290931134834
Epoch  470  loss  0.11934810590423062 correct 50
Epoch 470 took 1.9727835655212402 seconds
Average time per epoch: 2.193192275466433
Epoch  480  loss  0.37549892225824116 correct 50
Epoch 480 took 1.9799017906188965 seconds
Average time per epoch: 2.1924335495597855
Epoch  490  loss  0.21515525844422803 correct 50
Epoch 490 took 2.038817882537842 seconds
Average time per epoch: 2.1919304125177885
Average time per epoch: 2.1903444294929506
```

# Parallel Check Output

MAP

================================================================================
Parallel Accelerator Optimizing: Function tensor_map.<locals>.\_map,
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (164)

================================================================================

Parallel loop listing for Function tensor_map.<locals>.\_map, /Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (164)
-------------------------------------------------------------------------|loop #ID
def \_map( |
out: Storage, |
out_shape: Shape, |
out_strides: Strides, |
in_storage: Storage, |
in_shape: Shape, |
in_strides: Strides, |
) -> None: | # Direct storage access for aligned tensors |
if np.array_equal(out_shape, in_shape) and np.array_equal( |
out_strides, in_strides |
): |
for i in prange(len(out)):-----------------------------------| #2
out[i] = fn(in_storage[i]) |
return |
for i in prange(len(out)):---------------------------------------| #3
out_index = np.zeros(MAX_DIMS, np.int16)---------------------| #0
in_index = np.zeros(MAX_DIMS, np.int16)----------------------| #1
to_index(i, out_shape, out_index) |
broadcast_index(out_index, out_shape, in_shape, in_index) |
o = index_to_position(out_index, out_strides) |
j = index_to_position(in_index, in_strides) |
out[o] = fn(in_storage[j]) |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
+--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).

---

---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
+--0 --> rewritten as a serial loop

---

----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
+--0 (parallel)
+--1 (parallel)

---

------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
+--0 (serial, fused with loop(s): 1)

Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
of the larger parallel loop (#3).

---

---

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (180)
is hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
Allocation:: out_index = np.zeros(MAX_DIMS, np.int16) - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (181)
is hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
Allocation:: in_index = np.zeros(MAX_DIMS, np.int16) - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
Parallel Accelerator Optimizing: Function tensor_zip.<locals>.\_zip,
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (214)

================================================================================

Parallel loop listing for Function tensor_zip.<locals>.\_zip, /Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (214)
-----------------------------------------------------------------------|loop #ID
def \_zip( |
out: Storage, |
out_shape: Shape, |
out_strides: Strides, |
a_storage: Storage, |
a_shape: Shape, |
a_strides: Strides, |
b_storage: Storage, |
b_shape: Shape, |
b_strides: Strides, |
) -> None: |
if ( |
np.array_equal(out_shape, a_shape) |
and np.array_equal(out_shape, b_shape) |
and np.array_equal(out_strides, a_strides) |
and np.array_equal(out_strides, b_strides) |
): |
for i in prange(len(out)):---------------------------------| #7
out[i] = fn(a_storage[i], b_storage[i]) |
return |
for i in prange(len(out)):-------------------------------------| #8
out_index = np.zeros(MAX_DIMS, np.int16)-------------------| #4
a_index = np.zeros(MAX_DIMS, np.int16)---------------------| #5
b_index = np.zeros(MAX_DIMS, np.int16)---------------------| #6
to_index(i, out_shape, out_index) |
o = index_to_position(out_index, out_strides) |
broadcast_index(out_index, out_shape, a_shape, a_index) |
j = index_to_position(a_index, a_strides) |
broadcast_index(out_index, out_shape, b_shape, b_index) |
k = index_to_position(b_index, b_strides) |
out[o] = fn(a_storage[j], b_storage[k]) |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
+--5 (fused)
+--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).

---

---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
+--4 --> rewritten as a serial loop

---

----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
+--4 (parallel)
+--5 (parallel)
+--6 (parallel)

---

------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
+--4 (serial, fused with loop(s): 5, 6)

Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
of the larger parallel loop (#8).

---

---

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (235)
is hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
Allocation:: out_index = np.zeros(MAX_DIMS, np.int16) - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (236)
is hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
Allocation:: a_index = np.zeros(MAX_DIMS, np.int16) - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (237)
is hoisted out of the parallel loop labelled #8 (it will be performed before the
loop is executed and reused inside the loop):
Allocation:: b_index = np.zeros(MAX_DIMS, np.int16) - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
Parallel Accelerator Optimizing: Function tensor_reduce.<locals>.\_reduce,
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (270)

================================================================================

Parallel loop listing for Function tensor_reduce.<locals>.\_reduce, /Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (270)
---------------------------------------------------------------|loop #ID
def \_reduce( |
out: Storage, |
out_shape: Shape, |
out_strides: Strides, |
a_storage: Storage, |
a_shape: Shape, |
a_strides: Strides, |
reduce_dim: int, |
) -> None: |
reduce_size = a_shape[reduce_dim] |
| # Iterate over each element in the output tensor. |
for i in prange(len(out)):-----------------------------| #10
out_index = np.zeros(MAX_DIMS, np.int16)-----------| #9
to_index(i, out_shape, out_index) |
o = index_to_position(out_index, out_strides) |
for s in range(reduce_size): |
out_index[reduce_dim] = s |
j = index_to_position(out_index, a_strides) |
out[o] = fn(out[o], a_storage[j]) |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).

---

---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
+--9 --> rewritten as a serial loop

---

----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
+--9 (parallel)

---

------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
+--9 (serial)

Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).

---

---

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (283)
is hoisted out of the parallel loop labelled #10 (it will be performed before
the loop is executed and reused inside the loop):
Allocation:: out_index = np.zeros(MAX_DIMS, np.int16) - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
Parallel Accelerator Optimizing: Function \_tensor_matrix_multiply,
/Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (294)

================================================================================

Parallel loop listing for Function \_tensor_matrix_multiply, /Users/schaap/Documents/Cornell/mle/mod3-nickschaap/minitorch/fast_ops.py (294)
------------------------------------------------------------------------------------------|loop #ID
def \_tensor_matrix_multiply( |
out: Storage, |
out_shape: Shape, |
out_strides: Strides, |
a_storage: Storage, |
a_shape: Shape, |
a_strides: Strides, |
b_storage: Storage, |
b_shape: Shape, |
b_strides: Strides, |
) -> None: |
"""NUMBA tensor matrix multiply function. |
|
Should work for any tensor shapes that broadcast as long as |
|
`                                                                                  | 
    assert a_shape[-1] == b_shape[-2]                                                     | 
   ` |
|
Optimizations: |
|
_ Outer loop in parallel |
_ No index buffers or function calls |
_ Inner loop should have no global writes, 1 multiply. |
|
|
Args: |
---- |
out (Storage): storage for `out` tensor |
out_shape (Shape): shape for `out` tensor |
out_strides (Strides): strides for `out` tensor |
a_storage (Storage): storage for `a` tensor |
a_shape (Shape): shape for `a` tensor |
a_strides (Strides): strides for `a` tensor |
b_storage (Storage): storage for `b` tensor |
b_shape (Shape): shape for `b` tensor |
b_strides (Strides): strides for `b` tensor |
|
Returns: |
------- |
None : Fills in `out` |
|
""" |
a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0 |
b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0 |
| # Main loop in parallel |
for i in prange(out_shape[0]):--------------------------------------------------------| #11
for j in range(out_shape[1]): |
for k in range(out_shape[2]): | # Store accumulated value for (i,j,k) |
acc = 0.0 | # Inner loop - matrix multiply at position |
for l in range(a_shape[-1]): | # Get positions using strides |
a_pos = i _ a_batch_stride + j _ a_strides[1] + l _ a_strides[2] |
b_pos = i _ b_batch_stride + l _ b_strides[1] + k _ b_strides[2] |
acc += a_storage[a_pos] _ b_storage[b_pos] | # Set output position |
out_pos = i _ out_strides[0] + j _ out_strides[1] + k \* out_strides[2] |
out[out_pos] = acc |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).

---

## ----------------------------- Before Optimisation ------------------------------

------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.

---

---

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
