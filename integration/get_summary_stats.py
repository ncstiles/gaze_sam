res = """
(efficientvit) nicole@k9:~/gaze_sam/integration$ python3 combo.py 

~~~ ITER 1 with file ../base_imgs/gum.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.030138015747070312
prep encoder time: 1.0963821411132812
prep decoder time: 0.003520965576171875
prep encoder time: 0.001844167709350586
prep decoder time: 0.0018315315246582031
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 2.1457672119140625e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
cropped img embedding values: tensor([[-0.1012, -0.1158, -0.0677,  ..., -0.0169,  0.1105, -0.0514],
        [-0.0707, -0.0411,  0.0086,  ...,  0.0063, -0.1133, -0.0815],
        [-0.0755, -0.1127,  0.0452,  ..., -0.1107, -0.0719, -0.1144],
        ...,
        [-0.0507, -0.0664, -0.0659,  ..., -0.1177, -0.1163, -0.1231],
        [-0.0612, -0.0710, -0.0698,  ..., -0.1144, -0.1143, -0.1208],
        [-0.0893, -0.0820, -0.0820,  ..., -0.1036, -0.1001, -0.0932]],
       device='cuda:0')
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.027252912521362305
			point preprocessing time: 4.76837158203125e-07
				batch preprocess time: 0.0002961158752441406
				BATCH DECODER TIME: 0.004830121994018555
					convert to MaskData class: 8.392333984375e-05
					iou filtering time: 0.019517898559570312
					stability score filtering time: 0.0019707679748535156
					thresholding time: 0.001255035400390625
					box filtering time: 0.0009000301361083984
					mask uncrop time: 5.4836273193359375e-06
					rle compression time: 3.0994415283203125e-06
				batch filtering time: 0.023736238479614258
			batch process time: 0.02894878387451172
num iou preds before nms: torch.Size([29])
			batch nms time: 0.0010917186737060547
num iou preds after nms: torch.Size([13])
			uncrop time: 0.00014734268188476562
		crop process time: 0.058283090591430664
		duplicate crop removal time: 0.004057884216308594
mask data segmentations len: 13
	mask generation time: 0.06240487098693848
	postprocess time: 7.152557373046875e-07
	rle encoding time: 8.821487426757812e-06
	write MaskData: 0.00014519691467285156
number of bounding boxes: 18


~ extracting one mask ~
num anns: 13
img.shape: (720, 1280, 3)
get best max: 1700174648.6249306
find intersection point: 2.384185791015625e-07
set mask: 0.0026001930236816406
draw marker: 4.410743713378906e-05
draw line mask + best bounding box: 2.2411346435546875e-05

encoder/decoder priming run: 1.615330457687378
all gaze engines priming run: 0.10860657691955566
yolo priming run: 0.3915884494781494

load img: 0.06743669509887695
resize img: 2.116992235183716
generate masks: 0.06263947486877441
detect face (primed): 0.0026445388793945312
smooth + extract face (primed): 5.650520324707031e-05
detect landmark (primed): 0.0008418560028076172
smooth landmark (primed): 0.0005903244018554688
detect gaze (primed): 0.0034942626953125
smooth gaze (primed): 1.4543533325195312e-05
visualize gaze: 0.0007135868072509766
create plots: 6.9141387939453125e-06
get gaze mask: 0.0003800392150878906
prep yolo img: 0.0032224655151367188
yolo pred: 0.0028214454650878906
total yolo: 0.006043910980224609
draw and get yolo boxes: 0.0033063888549804688
segment one mask: 0.004163503646850586

display image: 0.017602205276489258
save to file (out/1700174658.4415767.png): 0.7180778980255127
non-load total: 0.08490109443664551
load total: 3.917043447494507


~~~ ITER 2 with file ../base_imgs/help.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.021310091018676758
prep encoder time: 0.004526853561401367
prep decoder time: 0.0029556751251220703
prep encoder time: 0.0019826889038085938
prep decoder time: 0.0018410682678222656
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.9073486328125e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
cropped img embedding values: tensor([[-0.0076, -0.1185, -0.1037,  ..., -0.0944, -0.1256, -0.0273],
        [ 0.0220,  0.0257,  0.0814,  ..., -0.0070, -0.1259, -0.0202],
        [ 0.0023, -0.0734,  0.0801,  ..., -0.0229, -0.1590, -0.0698],
        ...,
        [-0.0528, -0.0674, -0.0662,  ..., -0.1196, -0.1184, -0.1256],
        [-0.0641, -0.0721, -0.0702,  ..., -0.1165, -0.1166, -0.1239],
        [-0.0928, -0.0839, -0.0835,  ..., -0.1057, -0.1021, -0.0966]],
       device='cuda:0')
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.043680429458618164
			point preprocessing time: 2.384185791015625e-07
				batch preprocess time: 0.00018167495727539062
				BATCH DECODER TIME: 0.0030367374420166016
					convert to MaskData class: 5.435943603515625e-05
					iou filtering time: 0.021114349365234375
					stability score filtering time: 0.0025687217712402344
					thresholding time: 0.0003991127014160156
					box filtering time: 0.0006058216094970703
					mask uncrop time: 3.0994415283203125e-06
					rle compression time: 2.6226043701171875e-06
				batch filtering time: 0.02474808692932129
			batch process time: 0.028061628341674805
num iou preds before nms: torch.Size([47])
			batch nms time: 0.000591278076171875
num iou preds after nms: torch.Size([6])
			uncrop time: 0.00011301040649414062
		crop process time: 0.07306861877441406
		duplicate crop removal time: 0.0006835460662841797
mask data segmentations len: 6
	mask generation time: 0.07380294799804688
	postprocess time: 7.152557373046875e-07
	rle encoding time: 9.298324584960938e-06
	write MaskData: 8.368492126464844e-05
number of bounding boxes: 10


~ extracting one mask ~
num anns: 6
img.shape: (720, 1280, 3)
no box intersection
[   0.    0. 6132. 6217. 9629.  989.]
get best max: 1700174667.2447634
find intersection point: 2.384185791015625e-07
set mask: 0.002542257308959961
draw marker: 4.506111145019531e-05
draw line mask + best bounding box: 6.4373016357421875e-06

encoder/decoder priming run: 0.5038750171661377
all gaze engines priming run: 0.0964670181274414
yolo priming run: 0.3651864528656006

load img: 0.040154218673706055
resize img: 0.9663095474243164
generate masks: 0.07395577430725098
detect face (primed): 0.0024127960205078125
smooth + extract face (primed): 4.315376281738281e-05
detect landmark (primed): 0.0007925033569335938
smooth landmark (primed): 0.0005998611450195312
detect gaze (primed): 0.0033884048461914062
smooth gaze (primed): 1.1444091796875e-05
visualize gaze: 0.0006923675537109375
create plots: 6.9141387939453125e-06
get gaze mask: 0.00018477439880371094
prep yolo img: 0.004387617111206055
yolo pred: 0.002791166305541992
total yolo: 0.007178783416748047
draw and get yolo boxes: 0.003154754638671875
segment one mask: 0.0034613609313964844

display image: 0.0022957324981689453
save to file (out/1700174665.4152725.png): 0.8693747520446777
non-load total: 0.09589624404907227
load total: 0.7303318977355957


~~~ ITER 3 with file ../base_imgs/pen.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.020296335220336914
prep encoder time: 0.004570960998535156
prep decoder time: 0.0029435157775878906
prep encoder time: 0.0018506050109863281
prep decoder time: 0.0018739700317382812
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.6689300537109375e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
cropped img embedding values: tensor([[-0.1011, -0.0758, -0.0807,  ..., -0.0091, -0.0168,  0.1233],
        [-0.0549, -0.0470,  0.0114,  ..., -0.0685,  0.0290, -0.0815],
        [-0.0636, -0.0933,  0.0346,  ..., -0.0327, -0.1069, -0.0625],
        ...,
        [-0.0489, -0.0653, -0.0646,  ..., -0.1182, -0.1169, -0.1239],
        [-0.0595, -0.0695, -0.0682,  ..., -0.1151, -0.1148, -0.1217],
        [-0.0864, -0.0809, -0.0805,  ..., -0.1041, -0.1007, -0.0946]],
       device='cuda:0')
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.01822185516357422
			point preprocessing time: 4.76837158203125e-07
				batch preprocess time: 0.00018525123596191406
				BATCH DECODER TIME: 0.003425121307373047
					convert to MaskData class: 6.365776062011719e-05
					iou filtering time: 0.020872831344604492
					stability score filtering time: 0.0028078556060791016
					thresholding time: 0.0009119510650634766
					box filtering time: 0.0005469322204589844
					mask uncrop time: 3.5762786865234375e-06
					rle compression time: 2.384185791015625e-06
				batch filtering time: 0.02520918846130371
			batch process time: 0.028962135314941406
num iou preds before nms: torch.Size([61])
			batch nms time: 0.0005817413330078125
num iou preds after nms: torch.Size([12])
			uncrop time: 0.00013494491577148438
		crop process time: 0.04852724075317383
		duplicate crop removal time: 0.0011513233184814453
mask data segmentations len: 12
	mask generation time: 0.04972505569458008
	postprocess time: 4.76837158203125e-07
	rle encoding time: 7.152557373046875e-06
	write MaskData: 0.000125885009765625
number of bounding boxes: 18


~ extracting one mask ~
num anns: 12
img.shape: (720, 1280, 3)
get best max: 1700174669.0410585
find intersection point: 0.0
set mask: 0.005821943283081055
draw marker: 6.0558319091796875e-05
draw line mask + best bounding box: 2.2172927856445312e-05

encoder/decoder priming run: 0.5110957622528076
all gaze engines priming run: 0.09671258926391602
yolo priming run: 0.3663351535797119

load img: 0.07757115364074707
resize img: 0.9750065803527832
generate masks: 0.049916982650756836
detect face (primed): 0.0024366378784179688
smooth + extract face (primed): 4.2438507080078125e-05
detect landmark (primed): 0.0008296966552734375
smooth landmark (primed): 0.0006194114685058594
detect gaze (primed): 0.003421783447265625
smooth gaze (primed): 1.2636184692382812e-05
visualize gaze: 0.000598907470703125
create plots: 7.152557373046875e-06
get gaze mask: 0.00031447410583496094
prep yolo img: 0.003154277801513672
yolo pred: 0.002699613571166992
total yolo: 0.005853891372680664
draw and get yolo boxes: 0.0032722949981689453
segment one mask: 0.007495403289794922

display image: 0.002146482467651367
save to file (out/1700174668.1865973.png): 1.124297857284546
non-load total: 0.0748281478881836
load total: 0.7335286140441895


~~~ ITER 4 with file ../base_imgs/psycho.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.020585060119628906
prep encoder time: 0.00464177131652832
prep decoder time: 0.0031113624572753906
prep encoder time: 0.0018045902252197266
prep decoder time: 0.0018768310546875
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.9073486328125e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
cropped img embedding values: tensor([[-0.0631, -0.1142, -0.0939,  ..., -0.0019,  0.0752, -0.0294],
        [ 0.0044, -0.0160,  0.0820,  ...,  0.0516, -0.1061, -0.0576],
        [-0.0410, -0.0978,  0.0182,  ..., -0.1293, -0.0747, -0.0922],
        ...,
        [-0.0479, -0.0647, -0.0643,  ..., -0.1181, -0.1166, -0.1237],
        [-0.0577, -0.0687, -0.0679,  ..., -0.1150, -0.1146, -0.1212],
        [-0.0850, -0.0808, -0.0804,  ..., -0.1037, -0.1003, -0.0942]],
       device='cuda:0')
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.017790794372558594
			point preprocessing time: 2.384185791015625e-07
				batch preprocess time: 0.00016832351684570312
				BATCH DECODER TIME: 0.003390073776245117
					convert to MaskData class: 5.316734313964844e-05
					iou filtering time: 0.0207822322845459
					stability score filtering time: 0.002567291259765625
					thresholding time: 0.0004189014434814453
					box filtering time: 0.0007507801055908203
					mask uncrop time: 3.337860107421875e-06
					rle compression time: 2.6226043701171875e-06
				batch filtering time: 0.024578332901000977
			batch process time: 0.028222322463989258
num iou preds before nms: torch.Size([60])
			batch nms time: 0.0005183219909667969
num iou preds after nms: torch.Size([11])
			uncrop time: 0.0001277923583984375
		crop process time: 0.04726219177246094
		duplicate crop removal time: 0.0010693073272705078
mask data segmentations len: 11
	mask generation time: 0.04837942123413086
	postprocess time: 4.76837158203125e-07
	rle encoding time: 7.152557373046875e-06
	write MaskData: 0.00011038780212402344
number of bounding boxes: 13


~ extracting one mask ~
num anns: 11
img.shape: (720, 1280, 3)
get best max: 1700174673.0664592
find intersection point: 2.384185791015625e-07
set mask: 0.0058248043060302734
draw marker: 3.981590270996094e-05
draw line mask + best bounding box: 2.7418136596679688e-05

encoder/decoder priming run: 0.4990689754486084
all gaze engines priming run: 0.09655237197875977
yolo priming run: 0.36822032928466797

load img: 0.0756840705871582
resize img: 0.9647088050842285
generate masks: 0.04855036735534668
detect face (primed): 0.002434253692626953
smooth + extract face (primed): 4.291534423828125e-05
detect landmark (primed): 0.0007989406585693359
smooth landmark (primed): 0.0005791187286376953
detect gaze (primed): 0.0034842491149902344
smooth gaze (primed): 1.1920928955078125e-05
visualize gaze: 0.0005919933319091797
create plots: 6.9141387939453125e-06
get gaze mask: 0.00032973289489746094
prep yolo img: 0.004035234451293945
yolo pred: 0.002733469009399414
total yolo: 0.006768703460693359
draw and get yolo boxes: 0.003210306167602539
segment one mask: 0.00712275505065918

display image: 0.002177000045776367
save to file (out/1700174671.2400923.png): 1.366461992263794
non-load total: 0.07393813133239746
load total: 0.7184851169586182


~~~ ITER 5 with file ../base_imgs/workpls_v2.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.020930767059326172
prep encoder time: 0.004575490951538086
prep decoder time: 0.0030088424682617188
prep encoder time: 0.0018215179443359375
prep decoder time: 0.0018711090087890625
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.6689300537109375e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
cropped img embedding values: tensor([[-0.0539, -0.0417, -0.0188,  ..., -0.0824, -0.0756, -0.0572],
        [ 0.0002,  0.0026,  0.0541,  ..., -0.0517, -0.0408, -0.0274],
        [-0.0148, -0.0603,  0.0998,  ..., -0.0881, -0.0711, -0.0646],
        ...,
        [-0.0420, -0.0602, -0.0595,  ..., -0.1158, -0.1142, -0.1210],
        [-0.0533, -0.0642, -0.0627,  ..., -0.1122, -0.1121, -0.1180],
        [-0.0799, -0.0762, -0.0766,  ..., -0.1007, -0.0965, -0.0892]],
       device='cuda:0')
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.01786041259765625
			point preprocessing time: 4.76837158203125e-07
				batch preprocess time: 0.00015783309936523438
				BATCH DECODER TIME: 0.0034029483795166016
					convert to MaskData class: 4.9591064453125e-05
					iou filtering time: 0.021136999130249023
					stability score filtering time: 0.003933906555175781
					thresholding time: 0.0004062652587890625
					box filtering time: 0.0011661052703857422
					mask uncrop time: 3.5762786865234375e-06
					rle compression time: 2.86102294921875e-06
				batch filtering time: 0.026699304580688477
			batch process time: 0.030347347259521484
num iou preds before nms: torch.Size([104])
			batch nms time: 0.0005934238433837891
num iou preds after nms: torch.Size([9])
			uncrop time: 0.00011539459228515625
		crop process time: 0.049512624740600586
		duplicate crop removal time: 0.0009407997131347656
mask data segmentations len: 9
	mask generation time: 0.050505876541137695
	postprocess time: 7.152557373046875e-07
	rle encoding time: 7.62939453125e-06
	write MaskData: 9.512901306152344e-05
number of bounding boxes: 2


~ extracting one mask ~
num anns: 9
img.shape: (720, 1280, 3)
get best max: 1700174676.3859344
find intersection point: 0.0
set mask: 0.0024886131286621094
draw marker: 3.504753112792969e-05
draw line mask + best bounding box: 2.2649765014648438e-05

encoder/decoder priming run: 0.5017552375793457
all gaze engines priming run: 0.09638833999633789
yolo priming run: 0.37634849548339844

load img: 0.08002376556396484
resize img: 0.9753642082214355
generate masks: 0.050666093826293945
detect face (primed): 0.002481222152709961
smooth + extract face (primed): 4.458427429199219e-05
detect landmark (primed): 0.0008103847503662109
smooth landmark (primed): 0.0005998611450195312
detect gaze (primed): 0.0034630298614501953
smooth gaze (primed): 1.3828277587890625e-05
visualize gaze: 0.0006077289581298828
create plots: 6.9141387939453125e-06
get gaze mask: 0.0002548694610595703
prep yolo img: 0.0030112266540527344
yolo pred: 0.002740144729614258
total yolo: 0.005751371383666992
draw and get yolo boxes: 0.0030291080474853516
segment one mask: 0.0032682418823242188

display image: 0.002208709716796875
save to file (out/1700174674.5091987.png): 1.7466490268707275
non-load total: 0.07100319862365723
load total: 0.7540647983551025


~~~ ITER 6 with file ../base_imgs/zz.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.02066659927368164
prep encoder time: 0.004575490951538086
prep decoder time: 0.002959728240966797
prep encoder time: 0.0018019676208496094
prep decoder time: 0.0018281936645507812
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.6689300537109375e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
cropped img embedding values: tensor([[ 0.0265, -0.0816,  0.0267,  ..., -0.0457, -0.0007,  0.0368],
        [ 0.0069, -0.0404,  0.0218,  ..., -0.1111, -0.0613, -0.0206],
        [ 0.0189, -0.1010,  0.0568,  ...,  0.0591,  0.1163,  0.1045],
        ...,
        [-0.0546, -0.0686, -0.0677,  ..., -0.1197, -0.1185, -0.1257],
        [-0.0661, -0.0737, -0.0720,  ..., -0.1167, -0.1168, -0.1239],
        [-0.0944, -0.0852, -0.0849,  ..., -0.1061, -0.1031, -0.0970]],
       device='cuda:0')
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.017582178115844727
			point preprocessing time: 0.0
				batch preprocess time: 0.0001690387725830078
				BATCH DECODER TIME: 0.003373861312866211
					convert to MaskData class: 5.53131103515625e-05
					iou filtering time: 0.020608901977539062
					stability score filtering time: 0.002832651138305664
					thresholding time: 0.0004119873046875
					box filtering time: 0.0005934238433837891
					mask uncrop time: 3.337860107421875e-06
					rle compression time: 2.1457672119140625e-06
				batch filtering time: 0.024507761001586914
			batch process time: 0.02809906005859375
num iou preds before nms: torch.Size([47])
			batch nms time: 0.0005087852478027344
num iou preds after nms: torch.Size([7])
			uncrop time: 0.00010061264038085938
		crop process time: 0.04687142372131348
		duplicate crop removal time: 0.0007448196411132812
mask data segmentations len: 7
	mask generation time: 0.047669410705566406
	postprocess time: 4.76837158203125e-07
	rle encoding time: 5.7220458984375e-06
	write MaskData: 7.843971252441406e-05
number of bounding boxes: 11


~ extracting one mask ~
num anns: 7
img.shape: (720, 1280, 3)
get best max: 1700174672.0604396
find intersection point: 2.384185791015625e-07
set mask: 0.003924369812011719
draw marker: 3.4809112548828125e-05
draw line mask + best bounding box: 4.0531158447265625e-05

encoder/decoder priming run: 0.4932215213775635
all gaze engines priming run: 0.09637999534606934
yolo priming run: 0.38320088386535645

load img: 0.07373404502868652
resize img: 0.973618745803833
generate masks: 0.0478062629699707
detect face (primed): 0.002272367477416992
smooth + extract face (primed): 4.3392181396484375e-05
detect landmark (primed): 0.0007989406585693359
smooth landmark (primed): 0.0005805492401123047
detect gaze (primed): 0.0035529136657714844
smooth gaze (primed): 1.1920928955078125e-05
visualize gaze: 0.0006153583526611328
create plots: 7.152557373046875e-06
get gaze mask: 0.0002453327178955078
prep yolo img: 0.0029387474060058594
yolo pred: 0.002803802490234375
total yolo: 0.005742549896240234
draw and get yolo boxes: 0.003026723861694336
segment one mask: 0.004682064056396484

display image: 0.002134084701538086
save to file (out/1700174678.2077513.png): 1.8825299739837646
non-load total: 0.06939125061035156
load total: 0.740415096282959

(efficientvit) nicole@k9:~/gaze_sam/integration$ 
"""
total, encoder, decoder, iou, mask, yolo, segmentation = [], [], [], [], [], [], []

res = res.split("\n")
for line in res:
    if "non-load total:" in line:
        total.append(float(line.split(": ")[1]))
    elif "MASK ENCODER TIME: " in line:
        encoder.append(float(line.split(": ")[1]))
    elif "BATCH DECODER TIME: " in line:
        decoder.append(float(line.split(": ")[1]))
    elif "iou filtering time: " in line:
        iou.append(float(line.split(": ")[1]))
    elif "generate masks: " in line:
        mask.append(float(line.split(": ")[1]))
    elif "total yolo: " in line:
        yolo.append(float(line.split(": ")[1]))
    elif "segment one mask: " in line:
        segmentation.append(float(line.split(": ")[1]))
        
print("total time:\t\t\t",sum(total)/ len(total))
print("vit (encoder + decoder):\t\t\t", sum(mask)/len(mask))
print("encoder:\t\t\t", sum(encoder)/len(encoder))
print("decoder:\t\t\t", sum(decoder)/len(decoder))
print("iou:\t\t\t", sum(iou)/len(iou))
print("yolo:\t\t\t", sum(yolo)/len(yolo))
print("segment one mask:\t\t\t", sum(segmentation)/len(segmentation))
    
        
        
        
        
        
        
        
        