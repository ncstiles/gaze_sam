# # # # # # res = """
# # # # # # (efficientvit) nicole@k9:~/gaze_sam/integration$ python3 combo.py 

# # # # # # ~~~ ITER 1 with file ../base_imgs/gum.png ~~~
# # # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # # encoder preprocess time: 0.030138015747070312
# # # # # # prep encoder time: 1.0963821411132812
# # # # # # prep decoder time: 0.003520965576171875
# # # # # # prep encoder time: 0.001844167709350586
# # # # # # prep decoder time: 0.0018315315246582031
# # # # # # output shape: (2,)
# # # # # # Image Size: W=1280, H=720
# # # # # # output shape: (2,)
# # # # # # num crop boxes: 1
# # # # # # 			crop preprocess time: 2.1457672119140625e-06
# # # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # # cropped img embedding values: tensor([[-0.1012, -0.1158, -0.0677,  ..., -0.0169,  0.1105, -0.0514],
# # # # # #         [-0.0707, -0.0411,  0.0086,  ...,  0.0063, -0.1133, -0.0815],
# # # # # #         [-0.0755, -0.1127,  0.0452,  ..., -0.1107, -0.0719, -0.1144],
# # # # # #         ...,
# # # # # #         [-0.0507, -0.0664, -0.0659,  ..., -0.1177, -0.1163, -0.1231],
# # # # # #         [-0.0612, -0.0710, -0.0698,  ..., -0.1144, -0.1143, -0.1208],
# # # # # #         [-0.0893, -0.0820, -0.0820,  ..., -0.1036, -0.1001, -0.0932]],
# # # # # #        device='cuda:0')
# # # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # # 			MASK ENCODER TIME: 0.027252912521362305
# # # # # # 			point preprocessing time: 4.76837158203125e-07
# # # # # # 				batch preprocess time: 0.0002961158752441406
# # # # # # 				BATCH DECODER TIME: 0.004830121994018555
# # # # # # 					convert to MaskData class: 8.392333984375e-05
# # # # # # 					iou filtering time: 0.019517898559570312
# # # # # # 					stability score filtering time: 0.0019707679748535156
# # # # # # 					thresholding time: 0.001255035400390625
# # # # # # 					box filtering time: 0.0009000301361083984
# # # # # # 					mask uncrop time: 5.4836273193359375e-06
# # # # # # 					rle compression time: 3.0994415283203125e-06
# # # # # # 				batch filtering time: 0.023736238479614258
# # # # # # 			batch process time: 0.02894878387451172
# # # # # # num iou preds before nms: torch.Size([29])
# # # # # # 			batch nms time: 0.0010917186737060547
# # # # # # num iou preds after nms: torch.Size([13])
# # # # # # 			uncrop time: 0.00014734268188476562
# # # # # # 		crop process time: 0.058283090591430664
# # # # # # 		duplicate crop removal time: 0.004057884216308594
# # # # # # mask data segmentations len: 13
# # # # # # 	mask generation time: 0.06240487098693848
# # # # # # 	postprocess time: 7.152557373046875e-07
# # # # # # 	rle encoding time: 8.821487426757812e-06
# # # # # # 	write MaskData: 0.00014519691467285156
# # # # # # number of bounding boxes: 18


# # # # # # ~ extracting one mask ~
# # # # # # num anns: 13
# # # # # # img.shape: (720, 1280, 3)
# # # # # # get best max: 1700174648.6249306
# # # # # # find intersection point: 2.384185791015625e-07
# # # # # # set mask: 0.0026001930236816406
# # # # # # draw marker: 4.410743713378906e-05
# # # # # # draw line mask + best bounding box: 2.2411346435546875e-05

# # # # # # encoder/decoder priming run: 1.615330457687378
# # # # # # all gaze engines priming run: 0.10860657691955566
# # # # # # yolo priming run: 0.3915884494781494

# # # # # # load img: 0.06743669509887695
# # # # # # resize img: 2.116992235183716
# # # # # # generate masks: 0.06263947486877441
# # # # # # detect face (primed): 0.0026445388793945312
# # # # # # smooth + extract face (primed): 5.650520324707031e-05
# # # # # # detect landmark (primed): 0.0008418560028076172
# # # # # # smooth landmark (primed): 0.0005903244018554688
# # # # # # detect gaze (primed): 0.0034942626953125
# # # # # # smooth gaze (primed): 1.4543533325195312e-05
# # # # # # visualize gaze: 0.0007135868072509766
# # # # # # create plots: 6.9141387939453125e-06
# # # # # # get gaze mask: 0.0003800392150878906
# # # # # # prep yolo img: 0.0032224655151367188
# # # # # # yolo pred: 0.0028214454650878906
# # # # # # total yolo: 0.006043910980224609
# # # # # # draw and get yolo boxes: 0.0033063888549804688
# # # # # # segment one mask: 0.004163503646850586

# # # # # # display image: 0.017602205276489258
# # # # # # save to file (out/1700174658.4415767.png): 0.7180778980255127
# # # # # # non-load total: 0.08490109443664551
# # # # # # load total: 3.917043447494507


# # # # # # ~~~ ITER 2 with file ../base_imgs/help.png ~~~
# # # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # # encoder preprocess time: 0.021310091018676758
# # # # # # prep encoder time: 0.004526853561401367
# # # # # # prep decoder time: 0.0029556751251220703
# # # # # # prep encoder time: 0.0019826889038085938
# # # # # # prep decoder time: 0.0018410682678222656
# # # # # # output shape: (2,)
# # # # # # Image Size: W=1280, H=720
# # # # # # output shape: (2,)
# # # # # # num crop boxes: 1
# # # # # # 			crop preprocess time: 1.9073486328125e-06
# # # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # # cropped img embedding values: tensor([[-0.0076, -0.1185, -0.1037,  ..., -0.0944, -0.1256, -0.0273],
# # # # # #         [ 0.0220,  0.0257,  0.0814,  ..., -0.0070, -0.1259, -0.0202],
# # # # # #         [ 0.0023, -0.0734,  0.0801,  ..., -0.0229, -0.1590, -0.0698],
# # # # # #         ...,
# # # # # #         [-0.0528, -0.0674, -0.0662,  ..., -0.1196, -0.1184, -0.1256],
# # # # # #         [-0.0641, -0.0721, -0.0702,  ..., -0.1165, -0.1166, -0.1239],
# # # # # #         [-0.0928, -0.0839, -0.0835,  ..., -0.1057, -0.1021, -0.0966]],
# # # # # #        device='cuda:0')
# # # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # # 			MASK ENCODER TIME: 0.043680429458618164
# # # # # # 			point preprocessing time: 2.384185791015625e-07
# # # # # # 				batch preprocess time: 0.00018167495727539062
# # # # # # 				BATCH DECODER TIME: 0.0030367374420166016
# # # # # # 					convert to MaskData class: 5.435943603515625e-05
# # # # # # 					iou filtering time: 0.021114349365234375
# # # # # # 					stability score filtering time: 0.0025687217712402344
# # # # # # 					thresholding time: 0.0003991127014160156
# # # # # # 					box filtering time: 0.0006058216094970703
# # # # # # 					mask uncrop time: 3.0994415283203125e-06
# # # # # # 					rle compression time: 2.6226043701171875e-06
# # # # # # 				batch filtering time: 0.02474808692932129
# # # # # # 			batch process time: 0.028061628341674805
# # # # # # num iou preds before nms: torch.Size([47])
# # # # # # 			batch nms time: 0.000591278076171875
# # # # # # num iou preds after nms: torch.Size([6])
# # # # # # 			uncrop time: 0.00011301040649414062
# # # # # # 		crop process time: 0.07306861877441406
# # # # # # 		duplicate crop removal time: 0.0006835460662841797
# # # # # # mask data segmentations len: 6
# # # # # # 	mask generation time: 0.07380294799804688
# # # # # # 	postprocess time: 7.152557373046875e-07
# # # # # # 	rle encoding time: 9.298324584960938e-06
# # # # # # 	write MaskData: 8.368492126464844e-05
# # # # # # number of bounding boxes: 10


# # # # # # ~ extracting one mask ~
# # # # # # num anns: 6
# # # # # # img.shape: (720, 1280, 3)
# # # # # # no box intersection
# # # # # # [   0.    0. 6132. 6217. 9629.  989.]
# # # # # # get best max: 1700174667.2447634
# # # # # # find intersection point: 2.384185791015625e-07
# # # # # # set mask: 0.002542257308959961
# # # # # # draw marker: 4.506111145019531e-05
# # # # # # draw line mask + best bounding box: 6.4373016357421875e-06

# # # # # # encoder/decoder priming run: 0.5038750171661377
# # # # # # all gaze engines priming run: 0.0964670181274414
# # # # # # yolo priming run: 0.3651864528656006

# # # # # # load img: 0.040154218673706055
# # # # # # resize img: 0.9663095474243164
# # # # # # generate masks: 0.07395577430725098
# # # # # # detect face (primed): 0.0024127960205078125
# # # # # # smooth + extract face (primed): 4.315376281738281e-05
# # # # # # detect landmark (primed): 0.0007925033569335938
# # # # # # smooth landmark (primed): 0.0005998611450195312
# # # # # # detect gaze (primed): 0.0033884048461914062
# # # # # # smooth gaze (primed): 1.1444091796875e-05
# # # # # # visualize gaze: 0.0006923675537109375
# # # # # # create plots: 6.9141387939453125e-06
# # # # # # get gaze mask: 0.00018477439880371094
# # # # # # prep yolo img: 0.004387617111206055
# # # # # # yolo pred: 0.002791166305541992
# # # # # # total yolo: 0.007178783416748047
# # # # # # draw and get yolo boxes: 0.003154754638671875
# # # # # # segment one mask: 0.0034613609313964844

# # # # # # display image: 0.0022957324981689453
# # # # # # save to file (out/1700174665.4152725.png): 0.8693747520446777
# # # # # # non-load total: 0.09589624404907227
# # # # # # load total: 0.7303318977355957


# # # # # # ~~~ ITER 3 with file ../base_imgs/pen.png ~~~
# # # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # # encoder preprocess time: 0.020296335220336914
# # # # # # prep encoder time: 0.004570960998535156
# # # # # # prep decoder time: 0.0029435157775878906
# # # # # # prep encoder time: 0.0018506050109863281
# # # # # # prep decoder time: 0.0018739700317382812
# # # # # # output shape: (2,)
# # # # # # Image Size: W=1280, H=720
# # # # # # output shape: (2,)
# # # # # # num crop boxes: 1
# # # # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # # cropped img embedding values: tensor([[-0.1011, -0.0758, -0.0807,  ..., -0.0091, -0.0168,  0.1233],
# # # # # #         [-0.0549, -0.0470,  0.0114,  ..., -0.0685,  0.0290, -0.0815],
# # # # # #         [-0.0636, -0.0933,  0.0346,  ..., -0.0327, -0.1069, -0.0625],
# # # # # #         ...,
# # # # # #         [-0.0489, -0.0653, -0.0646,  ..., -0.1182, -0.1169, -0.1239],
# # # # # #         [-0.0595, -0.0695, -0.0682,  ..., -0.1151, -0.1148, -0.1217],
# # # # # #         [-0.0864, -0.0809, -0.0805,  ..., -0.1041, -0.1007, -0.0946]],
# # # # # #        device='cuda:0')
# # # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # # 			MASK ENCODER TIME: 0.01822185516357422
# # # # # # 			point preprocessing time: 4.76837158203125e-07
# # # # # # 				batch preprocess time: 0.00018525123596191406
# # # # # # 				BATCH DECODER TIME: 0.003425121307373047
# # # # # # 					convert to MaskData class: 6.365776062011719e-05
# # # # # # 					iou filtering time: 0.020872831344604492
# # # # # # 					stability score filtering time: 0.0028078556060791016
# # # # # # 					thresholding time: 0.0009119510650634766
# # # # # # 					box filtering time: 0.0005469322204589844
# # # # # # 					mask uncrop time: 3.5762786865234375e-06
# # # # # # 					rle compression time: 2.384185791015625e-06
# # # # # # 				batch filtering time: 0.02520918846130371
# # # # # # 			batch process time: 0.028962135314941406
# # # # # # num iou preds before nms: torch.Size([61])
# # # # # # 			batch nms time: 0.0005817413330078125
# # # # # # num iou preds after nms: torch.Size([12])
# # # # # # 			uncrop time: 0.00013494491577148438
# # # # # # 		crop process time: 0.04852724075317383
# # # # # # 		duplicate crop removal time: 0.0011513233184814453
# # # # # # mask data segmentations len: 12
# # # # # # 	mask generation time: 0.04972505569458008
# # # # # # 	postprocess time: 4.76837158203125e-07
# # # # # # 	rle encoding time: 7.152557373046875e-06
# # # # # # 	write MaskData: 0.000125885009765625
# # # # # # number of bounding boxes: 18


# # # # # # ~ extracting one mask ~
# # # # # # num anns: 12
# # # # # # img.shape: (720, 1280, 3)
# # # # # # get best max: 1700174669.0410585
# # # # # # find intersection point: 0.0
# # # # # # set mask: 0.005821943283081055
# # # # # # draw marker: 6.0558319091796875e-05
# # # # # # draw line mask + best bounding box: 2.2172927856445312e-05

# # # # # # encoder/decoder priming run: 0.5110957622528076
# # # # # # all gaze engines priming run: 0.09671258926391602
# # # # # # yolo priming run: 0.3663351535797119

# # # # # # load img: 0.07757115364074707
# # # # # # resize img: 0.9750065803527832
# # # # # # generate masks: 0.049916982650756836
# # # # # # detect face (primed): 0.0024366378784179688
# # # # # # smooth + extract face (primed): 4.2438507080078125e-05
# # # # # # detect landmark (primed): 0.0008296966552734375
# # # # # # smooth landmark (primed): 0.0006194114685058594
# # # # # # detect gaze (primed): 0.003421783447265625
# # # # # # smooth gaze (primed): 1.2636184692382812e-05
# # # # # # visualize gaze: 0.000598907470703125
# # # # # # create plots: 7.152557373046875e-06
# # # # # # get gaze mask: 0.00031447410583496094
# # # # # # prep yolo img: 0.003154277801513672
# # # # # # yolo pred: 0.002699613571166992
# # # # # # total yolo: 0.005853891372680664
# # # # # # draw and get yolo boxes: 0.0032722949981689453
# # # # # # segment one mask: 0.007495403289794922

# # # # # # display image: 0.002146482467651367
# # # # # # save to file (out/1700174668.1865973.png): 1.124297857284546
# # # # # # non-load total: 0.0748281478881836
# # # # # # load total: 0.7335286140441895


# # # # # # ~~~ ITER 4 with file ../base_imgs/psycho.png ~~~
# # # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # # encoder preprocess time: 0.020585060119628906
# # # # # # prep encoder time: 0.00464177131652832
# # # # # # prep decoder time: 0.0031113624572753906
# # # # # # prep encoder time: 0.0018045902252197266
# # # # # # prep decoder time: 0.0018768310546875
# # # # # # output shape: (2,)
# # # # # # Image Size: W=1280, H=720
# # # # # # output shape: (2,)
# # # # # # num crop boxes: 1
# # # # # # 			crop preprocess time: 1.9073486328125e-06
# # # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # # cropped img embedding values: tensor([[-0.0631, -0.1142, -0.0939,  ..., -0.0019,  0.0752, -0.0294],
# # # # # #         [ 0.0044, -0.0160,  0.0820,  ...,  0.0516, -0.1061, -0.0576],
# # # # # #         [-0.0410, -0.0978,  0.0182,  ..., -0.1293, -0.0747, -0.0922],
# # # # # #         ...,
# # # # # #         [-0.0479, -0.0647, -0.0643,  ..., -0.1181, -0.1166, -0.1237],
# # # # # #         [-0.0577, -0.0687, -0.0679,  ..., -0.1150, -0.1146, -0.1212],
# # # # # #         [-0.0850, -0.0808, -0.0804,  ..., -0.1037, -0.1003, -0.0942]],
# # # # # #        device='cuda:0')
# # # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # # 			MASK ENCODER TIME: 0.017790794372558594
# # # # # # 			point preprocessing time: 2.384185791015625e-07
# # # # # # 				batch preprocess time: 0.00016832351684570312
# # # # # # 				BATCH DECODER TIME: 0.003390073776245117
# # # # # # 					convert to MaskData class: 5.316734313964844e-05
# # # # # # 					iou filtering time: 0.0207822322845459
# # # # # # 					stability score filtering time: 0.002567291259765625
# # # # # # 					thresholding time: 0.0004189014434814453
# # # # # # 					box filtering time: 0.0007507801055908203
# # # # # # 					mask uncrop time: 3.337860107421875e-06
# # # # # # 					rle compression time: 2.6226043701171875e-06
# # # # # # 				batch filtering time: 0.024578332901000977
# # # # # # 			batch process time: 0.028222322463989258
# # # # # # num iou preds before nms: torch.Size([60])
# # # # # # 			batch nms time: 0.0005183219909667969
# # # # # # num iou preds after nms: torch.Size([11])
# # # # # # 			uncrop time: 0.0001277923583984375
# # # # # # 		crop process time: 0.04726219177246094
# # # # # # 		duplicate crop removal time: 0.0010693073272705078
# # # # # # mask data segmentations len: 11
# # # # # # 	mask generation time: 0.04837942123413086
# # # # # # 	postprocess time: 4.76837158203125e-07
# # # # # # 	rle encoding time: 7.152557373046875e-06
# # # # # # 	write MaskData: 0.00011038780212402344
# # # # # # number of bounding boxes: 13


# # # # # # ~ extracting one mask ~
# # # # # # num anns: 11
# # # # # # img.shape: (720, 1280, 3)
# # # # # # get best max: 1700174673.0664592
# # # # # # find intersection point: 2.384185791015625e-07
# # # # # # set mask: 0.0058248043060302734
# # # # # # draw marker: 3.981590270996094e-05
# # # # # # draw line mask + best bounding box: 2.7418136596679688e-05

# # # # # # encoder/decoder priming run: 0.4990689754486084
# # # # # # all gaze engines priming run: 0.09655237197875977
# # # # # # yolo priming run: 0.36822032928466797

# # # # # # load img: 0.0756840705871582
# # # # # # resize img: 0.9647088050842285
# # # # # # generate masks: 0.04855036735534668
# # # # # # detect face (primed): 0.002434253692626953
# # # # # # smooth + extract face (primed): 4.291534423828125e-05
# # # # # # detect landmark (primed): 0.0007989406585693359
# # # # # # smooth landmark (primed): 0.0005791187286376953
# # # # # # detect gaze (primed): 0.0034842491149902344
# # # # # # smooth gaze (primed): 1.1920928955078125e-05
# # # # # # visualize gaze: 0.0005919933319091797
# # # # # # create plots: 6.9141387939453125e-06
# # # # # # get gaze mask: 0.00032973289489746094
# # # # # # prep yolo img: 0.004035234451293945
# # # # # # yolo pred: 0.002733469009399414
# # # # # # total yolo: 0.006768703460693359
# # # # # # draw and get yolo boxes: 0.003210306167602539
# # # # # # segment one mask: 0.00712275505065918

# # # # # # display image: 0.002177000045776367
# # # # # # save to file (out/1700174671.2400923.png): 1.366461992263794
# # # # # # non-load total: 0.07393813133239746
# # # # # # load total: 0.7184851169586182


# # # # # # ~~~ ITER 5 with file ../base_imgs/workpls_v2.png ~~~
# # # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # # encoder preprocess time: 0.020930767059326172
# # # # # # prep encoder time: 0.004575490951538086
# # # # # # prep decoder time: 0.0030088424682617188
# # # # # # prep encoder time: 0.0018215179443359375
# # # # # # prep decoder time: 0.0018711090087890625
# # # # # # output shape: (2,)
# # # # # # Image Size: W=1280, H=720
# # # # # # output shape: (2,)
# # # # # # num crop boxes: 1
# # # # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # # cropped img embedding values: tensor([[-0.0539, -0.0417, -0.0188,  ..., -0.0824, -0.0756, -0.0572],
# # # # # #         [ 0.0002,  0.0026,  0.0541,  ..., -0.0517, -0.0408, -0.0274],
# # # # # #         [-0.0148, -0.0603,  0.0998,  ..., -0.0881, -0.0711, -0.0646],
# # # # # #         ...,
# # # # # #         [-0.0420, -0.0602, -0.0595,  ..., -0.1158, -0.1142, -0.1210],
# # # # # #         [-0.0533, -0.0642, -0.0627,  ..., -0.1122, -0.1121, -0.1180],
# # # # # #         [-0.0799, -0.0762, -0.0766,  ..., -0.1007, -0.0965, -0.0892]],
# # # # # #        device='cuda:0')
# # # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # # 			MASK ENCODER TIME: 0.01786041259765625
# # # # # # 			point preprocessing time: 4.76837158203125e-07
# # # # # # 				batch preprocess time: 0.00015783309936523438
# # # # # # 				BATCH DECODER TIME: 0.0034029483795166016
# # # # # # 					convert to MaskData class: 4.9591064453125e-05
# # # # # # 					iou filtering time: 0.021136999130249023
# # # # # # 					stability score filtering time: 0.003933906555175781
# # # # # # 					thresholding time: 0.0004062652587890625
# # # # # # 					box filtering time: 0.0011661052703857422
# # # # # # 					mask uncrop time: 3.5762786865234375e-06
# # # # # # 					rle compression time: 2.86102294921875e-06
# # # # # # 				batch filtering time: 0.026699304580688477
# # # # # # 			batch process time: 0.030347347259521484
# # # # # # num iou preds before nms: torch.Size([104])
# # # # # # 			batch nms time: 0.0005934238433837891
# # # # # # num iou preds after nms: torch.Size([9])
# # # # # # 			uncrop time: 0.00011539459228515625
# # # # # # 		crop process time: 0.049512624740600586
# # # # # # 		duplicate crop removal time: 0.0009407997131347656
# # # # # # mask data segmentations len: 9
# # # # # # 	mask generation time: 0.050505876541137695
# # # # # # 	postprocess time: 7.152557373046875e-07
# # # # # # 	rle encoding time: 7.62939453125e-06
# # # # # # 	write MaskData: 9.512901306152344e-05
# # # # # # number of bounding boxes: 2


# # # # # # ~ extracting one mask ~
# # # # # # num anns: 9
# # # # # # img.shape: (720, 1280, 3)
# # # # # # get best max: 1700174676.3859344
# # # # # # find intersection point: 0.0
# # # # # # set mask: 0.0024886131286621094
# # # # # # draw marker: 3.504753112792969e-05
# # # # # # draw line mask + best bounding box: 2.2649765014648438e-05

# # # # # # encoder/decoder priming run: 0.5017552375793457
# # # # # # all gaze engines priming run: 0.09638833999633789
# # # # # # yolo priming run: 0.37634849548339844

# # # # # # load img: 0.08002376556396484
# # # # # # resize img: 0.9753642082214355
# # # # # # generate masks: 0.050666093826293945
# # # # # # detect face (primed): 0.002481222152709961
# # # # # # smooth + extract face (primed): 4.458427429199219e-05
# # # # # # detect landmark (primed): 0.0008103847503662109
# # # # # # smooth landmark (primed): 0.0005998611450195312
# # # # # # detect gaze (primed): 0.0034630298614501953
# # # # # # smooth gaze (primed): 1.3828277587890625e-05
# # # # # # visualize gaze: 0.0006077289581298828
# # # # # # create plots: 6.9141387939453125e-06
# # # # # # get gaze mask: 0.0002548694610595703
# # # # # # prep yolo img: 0.0030112266540527344
# # # # # # yolo pred: 0.002740144729614258
# # # # # # total yolo: 0.005751371383666992
# # # # # # draw and get yolo boxes: 0.0030291080474853516
# # # # # # segment one mask: 0.0032682418823242188

# # # # # # display image: 0.002208709716796875
# # # # # # save to file (out/1700174674.5091987.png): 1.7466490268707275
# # # # # # non-load total: 0.07100319862365723
# # # # # # load total: 0.7540647983551025


# # # # # # ~~~ ITER 6 with file ../base_imgs/zz.png ~~~
# # # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # # encoder preprocess time: 0.02066659927368164
# # # # # # prep encoder time: 0.004575490951538086
# # # # # # prep decoder time: 0.002959728240966797
# # # # # # prep encoder time: 0.0018019676208496094
# # # # # # prep decoder time: 0.0018281936645507812
# # # # # # output shape: (2,)
# # # # # # Image Size: W=1280, H=720
# # # # # # output shape: (2,)
# # # # # # num crop boxes: 1
# # # # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # # cropped img embedding values: tensor([[ 0.0265, -0.0816,  0.0267,  ..., -0.0457, -0.0007,  0.0368],
# # # # # #         [ 0.0069, -0.0404,  0.0218,  ..., -0.1111, -0.0613, -0.0206],
# # # # # #         [ 0.0189, -0.1010,  0.0568,  ...,  0.0591,  0.1163,  0.1045],
# # # # # #         ...,
# # # # # #         [-0.0546, -0.0686, -0.0677,  ..., -0.1197, -0.1185, -0.1257],
# # # # # #         [-0.0661, -0.0737, -0.0720,  ..., -0.1167, -0.1168, -0.1239],
# # # # # #         [-0.0944, -0.0852, -0.0849,  ..., -0.1061, -0.1031, -0.0970]],
# # # # # #        device='cuda:0')
# # # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # # 			MASK ENCODER TIME: 0.017582178115844727
# # # # # # 			point preprocessing time: 0.0
# # # # # # 				batch preprocess time: 0.0001690387725830078
# # # # # # 				BATCH DECODER TIME: 0.003373861312866211
# # # # # # 					convert to MaskData class: 5.53131103515625e-05
# # # # # # 					iou filtering time: 0.020608901977539062
# # # # # # 					stability score filtering time: 0.002832651138305664
# # # # # # 					thresholding time: 0.0004119873046875
# # # # # # 					box filtering time: 0.0005934238433837891
# # # # # # 					mask uncrop time: 3.337860107421875e-06
# # # # # # 					rle compression time: 2.1457672119140625e-06
# # # # # # 				batch filtering time: 0.024507761001586914
# # # # # # 			batch process time: 0.02809906005859375
# # # # # # num iou preds before nms: torch.Size([47])
# # # # # # 			batch nms time: 0.0005087852478027344
# # # # # # num iou preds after nms: torch.Size([7])
# # # # # # 			uncrop time: 0.00010061264038085938
# # # # # # 		crop process time: 0.04687142372131348
# # # # # # 		duplicate crop removal time: 0.0007448196411132812
# # # # # # mask data segmentations len: 7
# # # # # # 	mask generation time: 0.047669410705566406
# # # # # # 	postprocess time: 4.76837158203125e-07
# # # # # # 	rle encoding time: 5.7220458984375e-06
# # # # # # 	write MaskData: 7.843971252441406e-05
# # # # # # number of bounding boxes: 11


# # # # # # ~ extracting one mask ~
# # # # # # num anns: 7
# # # # # # img.shape: (720, 1280, 3)
# # # # # # get best max: 1700174672.0604396
# # # # # # find intersection point: 2.384185791015625e-07
# # # # # # set mask: 0.003924369812011719
# # # # # # draw marker: 3.4809112548828125e-05
# # # # # # draw line mask + best bounding box: 4.0531158447265625e-05

# # # # # # encoder/decoder priming run: 0.4932215213775635
# # # # # # all gaze engines priming run: 0.09637999534606934
# # # # # # yolo priming run: 0.38320088386535645

# # # # # # load img: 0.07373404502868652
# # # # # # resize img: 0.973618745803833
# # # # # # generate masks: 0.0478062629699707
# # # # # # detect face (primed): 0.002272367477416992
# # # # # # smooth + extract face (primed): 4.3392181396484375e-05
# # # # # # detect landmark (primed): 0.0007989406585693359
# # # # # # smooth landmark (primed): 0.0005805492401123047
# # # # # # detect gaze (primed): 0.0035529136657714844
# # # # # # smooth gaze (primed): 1.1920928955078125e-05
# # # # # # visualize gaze: 0.0006153583526611328
# # # # # # create plots: 7.152557373046875e-06
# # # # # # get gaze mask: 0.0002453327178955078
# # # # # # prep yolo img: 0.0029387474060058594
# # # # # # yolo pred: 0.002803802490234375
# # # # # # total yolo: 0.005742549896240234
# # # # # # draw and get yolo boxes: 0.003026723861694336
# # # # # # segment one mask: 0.004682064056396484

# # # # # # display image: 0.002134084701538086
# # # # # # save to file (out/1700174678.2077513.png): 1.8825299739837646
# # # # # # non-load total: 0.06939125061035156
# # # # # # load total: 0.740415096282959

# # # # # # (efficientvit) nicole@k9:~/gaze_sam/integration$ 
# # # # # # """

# # # # # res = """

# # # # # ~~~ ITER 1 with file ../base_imgs/gum.png ~~~
# # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # encoder preprocess time: 0.02265453338623047
# # # # # prep encoder time: 0.04839920997619629
# # # # # prep decoder time: 0.0012543201446533203
# # # # # iou access time: 2.384185791015625e-07
# # # # # low res mask access time: 0.0
# # # # # prep encoder time: 0.0009877681732177734
# # # # # prep decoder time: 0.0006744861602783203
# # # # # iou access time: 0.0
# # # # # low res mask access time: 4.76837158203125e-07
# # # # # output shape: (2,)
# # # # # Image Size: W=1280, H=720
# # # # # output shape: (2,)
# # # # # num crop boxes: 1
# # # # # 			crop preprocess time: 1.430511474609375e-06
# # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # 			MASK ENCODER TIME: 0.009286880493164062
# # # # # 			point preprocessing time: 7.152557373046875e-07
# # # # # 				batch preprocess time: 0.004601716995239258
# # # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # # iou predictions shape: torch.Size([32, 4])
# # # # # tensor([0.9309, 0.9393, 0.9948, 0.9984], device='cuda:0')
# # # # # tensor([[ -8.6821,  -9.1047,  -8.5663,  ..., -10.1893, -10.7126, -10.8576],
# # # # #         [ -8.5263, -10.5178,  -8.6683,  ..., -14.7895, -19.1897, -11.2758],
# # # # #         [ -8.5589,  -9.1298,  -8.9727,  ..., -10.8972, -14.7396, -10.7822],
# # # # #         ...,
# # # # #         [-11.5409, -19.2368, -13.0331,  ..., -15.8325, -12.7569, -13.6186],
# # # # #         [-17.4800, -16.9283, -19.7576,  ..., -13.1044, -15.9927, -13.2330],
# # # # #         [-12.2174, -18.3461, -12.5894,  ..., -15.3497, -13.0314, -12.9609]],
# # # # #        device='cuda:0')
# # # # # decoder running time: 0.0009686946868896484
# # # # # time to access iou predictions before postprocessing 0.018574237823486328
# # # # # time to access low res masks before postprocessing 0.0013535022735595703
# # # # # 1.3113021850585938e-05
# # # # # 4.0531158447265625e-06
# # # # # 7.390975952148438e-05
# # # # # 3.123283386230469e-05
# # # # # 				BATCH DECODER TIME: 0.023155927658081055
# # # # # done filtering iou
# # # # # done filtering stability score
# # # # # done filtering edges
# # # # # 					convert to MaskData class: 0.00018024444580078125
# # # # # 					keep mask access time: 7.510185241699219e-05
# # # # # 					iou filtering time: 0.0021762847900390625
# # # # # 					stability score filtering time: 0.002101898193359375
# # # # # 					thresholding time: 0.0013704299926757812
# # # # # 					box filtering time: 9.775161743164062e-06
# # # # # 					mask uncrop time: 3.0994415283203125e-06
# # # # # 					rle compression time: 8.106231689453125e-06
# # # # # 				batch filtering time: 0.0058498382568359375
# # # # # 			batch process time: 0.03371119499206543
# # # # # num iou preds before nms: torch.Size([29])
# # # # # 			batch nms time: 0.0015823841094970703
# # # # # num iou preds after nms: torch.Size([12])
# # # # # 			uncrop time: 0.0001327991485595703
# # # # # 		crop process time: 0.04566311836242676
# # # # # 		duplicate crop removal time: 0.004887104034423828
# # # # # mask data segmentations len: 12
# # # # # 	mask generation time: 0.050637245178222656
# # # # # 	postprocess time: 9.5367431640625e-07
# # # # # 	rle encoding time: 5.4836273193359375e-06
# # # # # 	write MaskData: 0.00016880035400390625
# # # # # number of bounding boxes: 18


# # # # # ~ extracting one mask ~
# # # # # num anns: 12
# # # # # img.shape: (720, 1280, 3)
# # # # # get best max: 1700424046.8428173
# # # # # find intersection point: 2.384185791015625e-07
# # # # # set mask: 0.0025572776794433594
# # # # # draw marker: 5.793571472167969e-05
# # # # # draw line mask + best bounding box: 2.4080276489257812e-05

# # # # # encoder/decoder priming run: 0.5729939937591553
# # # # # all gaze engines priming run: 0.11265420913696289
# # # # # yolo priming run: 1.0995545387268066

# # # # # load img: 0.06767940521240234
# # # # # resize img: 1.7868266105651855
# # # # # generate masks: 0.050890207290649414
# # # # # detect face (primed): 0.0022668838500976562
# # # # # smooth + extract face (primed): 4.553794860839844e-05
# # # # # detect landmark (primed): 0.000759124755859375
# # # # # smooth landmark (primed): 0.0005774497985839844
# # # # # detect gaze (primed): 0.003422975540161133
# # # # # smooth gaze (primed): 1.2636184692382812e-05
# # # # # visualize gaze: 0.0008420944213867188
# # # # # create plots: 6.67572021484375e-06
# # # # # get gaze mask: 0.0004284381866455078
# # # # # prep yolo img: 0.0019927024841308594
# # # # # yolo pred: 0.0014367103576660156
# # # # # total yolo: 0.003429412841796875
# # # # # draw and get yolo boxes: 0.0042002201080322266
# # # # # segment one mask: 0.004098653793334961

# # # # # display image: 0.024402379989624023
# # # # # save to file (out/quantized_yolo/1700424046.9069438.png): 0.7367722988128662
# # # # # non-load total: 0.07098674774169922
# # # # # load total: 14.013415813446045


# # # # # ~~~ ITER 2 with file ../base_imgs/help.png ~~~
# # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # encoder preprocess time: 0.020836591720581055
# # # # # prep encoder time: 0.0017228126525878906
# # # # # prep decoder time: 0.0007164478302001953
# # # # # iou access time: 2.384185791015625e-07
# # # # # low res mask access time: 0.0
# # # # # prep encoder time: 0.0008974075317382812
# # # # # prep decoder time: 0.00045228004455566406
# # # # # iou access time: 0.0
# # # # # low res mask access time: 0.0
# # # # # output shape: (2,)
# # # # # Image Size: W=1280, H=720
# # # # # output shape: (2,)
# # # # # num crop boxes: 1
# # # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # 			MASK ENCODER TIME: 0.009785890579223633
# # # # # 			point preprocessing time: 4.76837158203125e-07
# # # # # 				batch preprocess time: 0.004436016082763672
# # # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # # iou predictions shape: torch.Size([32, 4])
# # # # # tensor([0.9356, 0.9962, 0.9615, 0.9885], device='cuda:0')
# # # # # tensor([[ -7.0160,  -8.3375,  -7.6235,  ...,  -9.8531,  -8.8872,  -9.3785],
# # # # #         [ -6.8658, -12.2383,  -8.6802,  ..., -11.4151, -12.3284,  -9.5103],
# # # # #         [ -6.9508, -10.1973,  -8.2336,  ...,  -9.3862,  -9.9685,  -8.7662],
# # # # #         ...,
# # # # #         [-11.3395, -18.4510, -13.1705,  ..., -15.2834, -12.5159, -13.6508],
# # # # #         [-15.8862, -16.0607, -18.3657,  ..., -12.7847, -14.6716, -12.8057],
# # # # #         [-12.0189, -17.9818, -12.8231,  ..., -15.0133, -12.7326, -13.1078]],
# # # # #        device='cuda:0')
# # # # # decoder running time: 0.0012006759643554688
# # # # # time to access iou predictions before postprocessing 0.015514850616455078
# # # # # time to access low res masks before postprocessing 0.0011670589447021484
# # # # # 2.09808349609375e-05
# # # # # 4.291534423828125e-06
# # # # # 2.1219253540039062e-05
# # # # # 7.867813110351562e-06
# # # # # 				BATCH DECODER TIME: 0.01824212074279785
# # # # # done filtering iou
# # # # # done filtering stability score
# # # # # done filtering edges
# # # # # 					convert to MaskData class: 7.510185241699219e-05
# # # # # 					keep mask access time: 1.4543533325195312e-05
# # # # # 					iou filtering time: 0.0028967857360839844
# # # # # 					stability score filtering time: 0.0029125213623046875
# # # # # 					thresholding time: 0.0016164779663085938
# # # # # 					box filtering time: 9.059906005859375e-06
# # # # # 					mask uncrop time: 2.86102294921875e-06
# # # # # 					rle compression time: 1.9073486328125e-06
# # # # # 				batch filtering time: 0.0075147151947021484
# # # # # 			batch process time: 0.030347824096679688
# # # # # num iou preds before nms: torch.Size([60])
# # # # # 			batch nms time: 0.0007348060607910156
# # # # # num iou preds after nms: torch.Size([6])
# # # # # 			uncrop time: 0.00010538101196289062
# # # # # 		crop process time: 0.04161548614501953
# # # # # 		duplicate crop removal time: 0.00054168701171875
# # # # # mask data segmentations len: 6
# # # # # 	mask generation time: 0.04221034049987793
# # # # # 	postprocess time: 4.76837158203125e-07
# # # # # 	rle encoding time: 5.4836273193359375e-06
# # # # # 	write MaskData: 9.393692016601562e-05
# # # # # number of bounding boxes: 10


# # # # # ~ extracting one mask ~
# # # # # num anns: 6
# # # # # img.shape: (720, 1280, 3)
# # # # # no box intersection
# # # # # [   0. 6179.    0. 9621. 5891.  987.]
# # # # # get best max: 1700424066.2586424
# # # # # find intersection point: 2.384185791015625e-07
# # # # # set mask: 0.0025467872619628906
# # # # # draw marker: 4.601478576660156e-05
# # # # # draw line mask + best bounding box: 5.7220458984375e-06

# # # # # encoder/decoder priming run: 0.5028076171875
# # # # # all gaze engines priming run: 0.09452486038208008
# # # # # yolo priming run: 1.09228515625

# # # # # load img: 0.04040932655334473
# # # # # resize img: 1.6910245418548584
# # # # # generate masks: 0.04236745834350586
# # # # # detect face (primed): 0.0038390159606933594
# # # # # smooth + extract face (primed): 4.601478576660156e-05
# # # # # detect landmark (primed): 0.0009326934814453125
# # # # # smooth landmark (primed): 0.0005943775177001953
# # # # # detect gaze (primed): 0.003595590591430664
# # # # # smooth gaze (primed): 1.1444091796875e-05
# # # # # visualize gaze: 0.0007455348968505859
# # # # # create plots: 6.198883056640625e-06
# # # # # get gaze mask: 0.0002512931823730469
# # # # # prep yolo img: 0.0020084381103515625
# # # # # yolo pred: 0.0012989044189453125
# # # # # total yolo: 0.003307342529296875
# # # # # draw and get yolo boxes: 0.0039594173431396484
# # # # # segment one mask: 0.003689289093017578

# # # # # display image: 0.002625703811645508
# # # # # save to file (out/quantized_yolo/1700424063.6191363.png): 0.895561933517456
# # # # # non-load total: 0.06335163116455078
# # # # # load total: 0.8485937118530273


# # # # # ~~~ ITER 3 with file ../base_imgs/pen.png ~~~
# # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # encoder preprocess time: 0.020688533782958984
# # # # # prep encoder time: 0.0017170906066894531
# # # # # prep decoder time: 0.0008029937744140625
# # # # # iou access time: 4.76837158203125e-07
# # # # # low res mask access time: 0.0
# # # # # prep encoder time: 0.0008683204650878906
# # # # # prep decoder time: 0.00044655799865722656
# # # # # iou access time: 0.0
# # # # # low res mask access time: 2.384185791015625e-07
# # # # # output shape: (2,)
# # # # # Image Size: W=1280, H=720
# # # # # output shape: (2,)
# # # # # num crop boxes: 1
# # # # # 			crop preprocess time: 1.430511474609375e-06
# # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # 			MASK ENCODER TIME: 0.00966024398803711
# # # # # 			point preprocessing time: 7.152557373046875e-07
# # # # # 				batch preprocess time: 0.004380941390991211
# # # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # # iou predictions shape: torch.Size([32, 4])
# # # # # tensor([0.9372, 0.8541, 1.0003, 0.9865], device='cuda:0')
# # # # # tensor([[ -8.0574,  -8.3014,  -7.7984,  ...,  -8.8410,  -9.0426,  -9.1811],
# # # # #         [ -8.5147,  -9.9573,  -7.7391,  ..., -10.3093, -12.2576,  -9.1797],
# # # # #         [ -7.8883,  -8.3459,  -9.0293,  ...,  -8.3805,  -9.1023,  -8.3246],
# # # # #         ...,
# # # # #         [-11.7061, -19.4598, -13.2008,  ..., -15.2920, -12.2584, -13.5973],
# # # # #         [-17.6420, -16.8277, -19.6731,  ..., -12.9862, -15.2091, -12.9323],
# # # # #         [-12.7394, -18.8989, -13.0475,  ..., -15.2122, -12.4640, -13.1977]],
# # # # #        device='cuda:0')
# # # # # decoder running time: 0.0008563995361328125
# # # # # time to access iou predictions before postprocessing 0.015442848205566406
# # # # # time to access low res masks before postprocessing 0.0011754035949707031
# # # # # 1.3113021850585938e-05
# # # # # 4.291534423828125e-06
# # # # # 1.7881393432617188e-05
# # # # # 6.67572021484375e-06
# # # # # 				BATCH DECODER TIME: 0.017787694931030273
# # # # # done filtering iou
# # # # # done filtering stability score
# # # # # done filtering edges
# # # # # 					convert to MaskData class: 7.653236389160156e-05
# # # # # 					keep mask access time: 1.2874603271484375e-05
# # # # # 					iou filtering time: 0.002855062484741211
# # # # # 					stability score filtering time: 0.0028600692749023438
# # # # # 					thresholding time: 0.00043845176696777344
# # # # # 					box filtering time: 9.059906005859375e-06
# # # # # 					mask uncrop time: 2.86102294921875e-06
# # # # # 					rle compression time: 2.86102294921875e-06
# # # # # 				batch filtering time: 0.0062448978424072266
# # # # # 			batch process time: 0.02847146987915039
# # # # # num iou preds before nms: torch.Size([65])
# # # # # 			batch nms time: 0.0006377696990966797
# # # # # num iou preds after nms: torch.Size([12])
# # # # # 			uncrop time: 0.00011086463928222656
# # # # # 		crop process time: 0.03951001167297363
# # # # # 		duplicate crop removal time: 0.0009403228759765625
# # # # # mask data segmentations len: 12
# # # # # 	mask generation time: 0.04049968719482422
# # # # # 	postprocess time: 4.76837158203125e-07
# # # # # 	rle encoding time: 5.245208740234375e-06
# # # # # 	write MaskData: 0.0001239776611328125
# # # # # number of bounding boxes: 18


# # # # # ~ extracting one mask ~
# # # # # num anns: 12
# # # # # img.shape: (720, 1280, 3)
# # # # # get best max: 1700424068.923072
# # # # # find intersection point: 2.384185791015625e-07
# # # # # set mask: 0.0059888362884521484
# # # # # draw marker: 5.125999450683594e-05
# # # # # draw line mask + best bounding box: 2.2411346435546875e-05

# # # # # encoder/decoder priming run: 0.5385475158691406
# # # # # all gaze engines priming run: 0.0937492847442627
# # # # # yolo priming run: 1.0889463424682617

# # # # # load img: 0.07642865180969238
# # # # # resize img: 1.722738265991211
# # # # # generate masks: 0.04067707061767578
# # # # # detect face (primed): 0.0033643245697021484
# # # # # smooth + extract face (primed): 4.38690185546875e-05
# # # # # detect landmark (primed): 0.0007894039154052734
# # # # # smooth landmark (primed): 0.0005433559417724609
# # # # # detect gaze (primed): 0.0034334659576416016
# # # # # smooth gaze (primed): 1.2636184692382812e-05
# # # # # visualize gaze: 0.0006308555603027344
# # # # # create plots: 5.7220458984375e-06
# # # # # get gaze mask: 0.0003325939178466797
# # # # # prep yolo img: 0.0014982223510742188
# # # # # yolo pred: 0.001299142837524414
# # # # # total yolo: 0.002797365188598633
# # # # # draw and get yolo boxes: 0.004008054733276367
# # # # # segment one mask: 0.007664680480957031

# # # # # display image: 0.0022423267364501953
# # # # # save to file (out/quantized_yolo/1700424067.1957061.png): 1.1597874164581299
# # # # # non-load total: 0.06430959701538086
# # # # # load total: 0.8705577850341797


# # # # # ~~~ ITER 4 with file ../base_imgs/psycho.png ~~~
# # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # encoder preprocess time: 0.02073502540588379
# # # # # prep encoder time: 0.0017616748809814453
# # # # # prep decoder time: 0.0007841587066650391
# # # # # iou access time: 0.0
# # # # # low res mask access time: 2.384185791015625e-07
# # # # # prep encoder time: 0.0008726119995117188
# # # # # prep decoder time: 0.00046062469482421875
# # # # # iou access time: 0.0
# # # # # low res mask access time: 0.0
# # # # # output shape: (2,)
# # # # # Image Size: W=1280, H=720
# # # # # output shape: (2,)
# # # # # num crop boxes: 1
# # # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # 			MASK ENCODER TIME: 0.009652853012084961
# # # # # 			point preprocessing time: 7.152557373046875e-07
# # # # # 				batch preprocess time: 0.0044367313385009766
# # # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # # iou predictions shape: torch.Size([32, 4])
# # # # # tensor([0.9273, 0.9296, 0.9998, 0.9919], device='cuda:0')
# # # # # tensor([[ -8.4912,  -9.1068,  -8.5180,  ...,  -7.8927,  -9.0134,  -8.8053],
# # # # #         [ -7.9993, -12.9304,  -9.9381,  ...,  -9.6036, -11.0376,  -9.9365],
# # # # #         [ -7.7318,  -9.8774,  -8.8662,  ...,  -8.8792,  -9.2981,  -9.4876],
# # # # #         ...,
# # # # #         [-11.9294, -18.7491, -13.2263,  ..., -14.2644, -11.9080, -12.8588],
# # # # #         [-17.8201, -17.1532, -19.4006,  ..., -12.6998, -14.0653, -12.3572],
# # # # #         [-13.0063, -18.6890, -13.1739,  ..., -14.5281, -12.0609, -12.6349]],
# # # # #        device='cuda:0')
# # # # # decoder running time: 0.0008769035339355469
# # # # # time to access iou predictions before postprocessing 0.015523195266723633
# # # # # time to access low res masks before postprocessing 0.0011436939239501953
# # # # # 1.2159347534179688e-05
# # # # # 4.291534423828125e-06
# # # # # 1.811981201171875e-05
# # # # # 6.9141387939453125e-06
# # # # # 				BATCH DECODER TIME: 0.017899036407470703
# # # # # done filtering iou
# # # # # done filtering stability score
# # # # # done filtering edges
# # # # # 					convert to MaskData class: 7.486343383789062e-05
# # # # # 					keep mask access time: 1.3113021850585938e-05
# # # # # 					iou filtering time: 0.002796173095703125
# # # # # 					stability score filtering time: 0.0027518272399902344
# # # # # 					thresholding time: 0.0004551410675048828
# # # # # 					box filtering time: 8.821487426757812e-06
# # # # # 					mask uncrop time: 2.86102294921875e-06
# # # # # 					rle compression time: 2.384185791015625e-06
# # # # # 				batch filtering time: 0.006092071533203125
# # # # # 			batch process time: 0.028481721878051758
# # # # # num iou preds before nms: torch.Size([69])
# # # # # 			batch nms time: 0.0006558895111083984
# # # # # num iou preds after nms: torch.Size([13])
# # # # # 			uncrop time: 0.00011157989501953125
# # # # # 		crop process time: 0.039513349533081055
# # # # # 		duplicate crop removal time: 0.0009903907775878906
# # # # # mask data segmentations len: 13
# # # # # 	mask generation time: 0.04055309295654297
# # # # # 	postprocess time: 4.76837158203125e-07
# # # # # 	rle encoding time: 5.9604644775390625e-06
# # # # # 	write MaskData: 0.00012969970703125
# # # # # number of bounding boxes: 13


# # # # # ~ extracting one mask ~
# # # # # num anns: 13
# # # # # img.shape: (720, 1280, 3)
# # # # # get best max: 1700424073.771699
# # # # # find intersection point: 2.384185791015625e-07
# # # # # set mask: 0.006228923797607422
# # # # # draw marker: 4.887580871582031e-05
# # # # # draw line mask + best bounding box: 2.765655517578125e-05

# # # # # encoder/decoder priming run: 0.5064294338226318
# # # # # all gaze engines priming run: 0.09419918060302734
# # # # # yolo priming run: 1.0919270515441895

# # # # # load img: 0.07616186141967773
# # # # # resize img: 1.6938362121582031
# # # # # generate masks: 0.040741682052612305
# # # # # detect face (primed): 0.0033960342407226562
# # # # # smooth + extract face (primed): 4.291534423828125e-05
# # # # # detect landmark (primed): 0.0007903575897216797
# # # # # smooth landmark (primed): 0.0005509853363037109
# # # # # detect gaze (primed): 0.003369569778442383
# # # # # smooth gaze (primed): 1.1444091796875e-05
# # # # # visualize gaze: 0.0006394386291503906
# # # # # create plots: 5.7220458984375e-06
# # # # # get gaze mask: 0.00033736228942871094
# # # # # prep yolo img: 0.0016360282897949219
# # # # # yolo pred: 0.001283407211303711
# # # # # total yolo: 0.002919435501098633
# # # # # draw and get yolo boxes: 0.003987550735473633
# # # # # segment one mask: 0.007771730422973633

# # # # # display image: 0.0023360252380371094
# # # # # save to file (out/quantized_yolo/1700424071.1244743.png): 1.3589627742767334
# # # # # non-load total: 0.06457185745239258
# # # # # load total: 0.8195161819458008


# # # # # ~~~ ITER 5 with file ../base_imgs/workpls_v2.png ~~~
# # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # encoder preprocess time: 0.021640539169311523
# # # # # prep encoder time: 0.0017125606536865234
# # # # # prep decoder time: 0.0007982254028320312
# # # # # iou access time: 4.76837158203125e-07
# # # # # low res mask access time: 0.0
# # # # # prep encoder time: 0.0008752346038818359
# # # # # prep decoder time: 0.00046062469482421875
# # # # # iou access time: 0.0
# # # # # low res mask access time: 2.384185791015625e-07
# # # # # output shape: (2,)
# # # # # Image Size: W=1280, H=720
# # # # # output shape: (2,)
# # # # # num crop boxes: 1
# # # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # 			MASK ENCODER TIME: 0.009545326232910156
# # # # # 			point preprocessing time: 2.384185791015625e-07
# # # # # 				batch preprocess time: 0.004525423049926758
# # # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # # iou predictions shape: torch.Size([32, 4])
# # # # # tensor([0.8442, 0.9641, 0.9934, 1.0000], device='cuda:0')
# # # # # tensor([[ -7.3666,  -7.6250,  -7.0991,  ...,  -7.7910,  -8.1833,  -7.9219],
# # # # #         [ -7.1297,  -8.4599,  -6.9819,  ...,  -9.7629, -10.5244, -10.0282],
# # # # #         [ -7.8415,  -8.2238,  -7.8710,  ...,  -7.8803,  -8.3298,  -8.5783],
# # # # #         ...,
# # # # #         [ -9.9370, -15.3063, -11.1078,  ..., -11.4604, -10.0422, -10.1142],
# # # # #         [-14.3952, -14.0464, -16.0325,  ..., -10.1470, -11.3168,  -9.7674],
# # # # #         [-10.6241, -15.1760, -10.9802,  ..., -11.7795, -10.1965, -10.0544]],
# # # # #        device='cuda:0')
# # # # # decoder running time: 0.0008695125579833984
# # # # # time to access iou predictions before postprocessing 0.01567363739013672
# # # # # time to access low res masks before postprocessing 0.0011363029479980469
# # # # # 1.2874603271484375e-05
# # # # # 4.76837158203125e-06
# # # # # 1.7881393432617188e-05
# # # # # 6.67572021484375e-06
# # # # # 				BATCH DECODER TIME: 0.017991065979003906
# # # # # done filtering iou
# # # # # done filtering stability score
# # # # # done filtering edges
# # # # # 					convert to MaskData class: 7.224082946777344e-05
# # # # # 					keep mask access time: 1.33514404296875e-05
# # # # # 					iou filtering time: 0.0031137466430664062
# # # # # 					stability score filtering time: 0.0038776397705078125
# # # # # 					thresholding time: 0.0004200935363769531
# # # # # 					box filtering time: 8.821487426757812e-06
# # # # # 					mask uncrop time: 2.86102294921875e-06
# # # # # 					rle compression time: 2.6226043701171875e-06
# # # # # 				batch filtering time: 0.007498025894165039
# # # # # 			batch process time: 0.030078411102294922
# # # # # num iou preds before nms: torch.Size([101])
# # # # # 			batch nms time: 0.0009624958038330078
# # # # # num iou preds after nms: torch.Size([8])
# # # # # 			uncrop time: 0.00010395050048828125
# # # # # 		crop process time: 0.041326045989990234
# # # # # 		duplicate crop removal time: 0.0007102489471435547
# # # # # mask data segmentations len: 8
# # # # # 	mask generation time: 0.042084693908691406
# # # # # 	postprocess time: 2.384185791015625e-07
# # # # # 	rle encoding time: 5.9604644775390625e-06
# # # # # 	write MaskData: 8.726119995117188e-05
# # # # # number of bounding boxes: 2


# # # # # ~ extracting one mask ~
# # # # # num anns: 8
# # # # # img.shape: (720, 1280, 3)
# # # # # get best max: 1700424077.8385468
# # # # # find intersection point: 0.0
# # # # # set mask: 0.0025000572204589844
# # # # # draw marker: 3.933906555175781e-05
# # # # # draw line mask + best bounding box: 2.8848648071289062e-05

# # # # # encoder/decoder priming run: 0.5060861110687256
# # # # # all gaze engines priming run: 0.0943915843963623
# # # # # yolo priming run: 1.0978515148162842

# # # # # load img: 0.07965445518493652
# # # # # resize img: 1.6992003917694092
# # # # # generate masks: 0.04222536087036133
# # # # # detect face (primed): 0.003434896469116211
# # # # # smooth + extract face (primed): 4.267692565917969e-05
# # # # # detect landmark (primed): 0.0007991790771484375
# # # # # smooth landmark (primed): 0.0005817413330078125
# # # # # detect gaze (primed): 0.0033903121948242188
# # # # # smooth gaze (primed): 1.33514404296875e-05
# # # # # visualize gaze: 0.0006666183471679688
# # # # # create plots: 6.4373016357421875e-06
# # # # # get gaze mask: 0.0002465248107910156
# # # # # prep yolo img: 0.0013227462768554688
# # # # # yolo pred: 0.0012981891632080078
# # # # # total yolo: 0.0026209354400634766
# # # # # draw and get yolo boxes: 0.003904581069946289
# # # # # segment one mask: 0.006760835647583008

# # # # # display image: 0.002296924591064453
# # # # # save to file (out/quantized_yolo/1700424075.1739895.png): 1.756664752960205
# # # # # non-load total: 0.06470012664794922
# # # # # load total: 0.8241417407989502


# # # # # ~~~ ITER 6 with file ../base_imgs/zz.png ~~~
# # # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # # encoder preprocess time: 0.020522356033325195
# # # # # prep encoder time: 0.0016446113586425781
# # # # # prep decoder time: 0.0007522106170654297
# # # # # iou access time: 4.76837158203125e-07
# # # # # low res mask access time: 2.384185791015625e-07
# # # # # prep encoder time: 0.0008919239044189453
# # # # # prep decoder time: 0.00045037269592285156
# # # # # iou access time: 0.0
# # # # # low res mask access time: 2.384185791015625e-07
# # # # # output shape: (2,)
# # # # # Image Size: W=1280, H=720
# # # # # output shape: (2,)
# # # # # num crop boxes: 1
# # # # # 			crop preprocess time: 1.430511474609375e-06
# # # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # # 			MASK ENCODER TIME: 0.009821891784667969
# # # # # 			point preprocessing time: 4.76837158203125e-07
# # # # # 				batch preprocess time: 0.004532814025878906
# # # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # # iou predictions shape: torch.Size([32, 4])
# # # # # tensor([0.8999, 0.9947, 0.9966, 0.9902], device='cuda:0')
# # # # # tensor([[ -7.6425,  -8.1124,  -8.3656,  ...,  -9.7908,  -9.3453, -10.3772],
# # # # #         [ -6.6389,  -9.6652,  -8.0147,  ..., -11.2941, -11.3401, -11.9898],
# # # # #         [ -7.3191,  -9.0333,  -8.6178,  ...,  -9.0132,  -9.1758,  -9.8109],
# # # # #         ...,
# # # # #         [-11.2575, -18.9012, -12.8749,  ..., -15.8234, -13.1052, -13.7998],
# # # # #         [-16.4673, -16.1071, -19.0224,  ..., -12.8880, -14.8513, -12.8635],
# # # # #         [-12.0691, -18.3094, -12.8685,  ..., -15.8430, -13.4844, -13.4991]],
# # # # #        device='cuda:0')
# # # # # decoder running time: 0.0008828639984130859
# # # # # time to access iou predictions before postprocessing 0.015610694885253906
# # # # # time to access low res masks before postprocessing 0.0011355876922607422
# # # # # 1.2159347534179688e-05
# # # # # 4.291534423828125e-06
# # # # # 1.7881393432617188e-05
# # # # # 6.198883056640625e-06
# # # # # 				BATCH DECODER TIME: 0.017944812774658203
# # # # # done filtering iou
# # # # # done filtering stability score
# # # # # done filtering edges
# # # # # 					convert to MaskData class: 7.796287536621094e-05
# # # # # 					keep mask access time: 1.430511474609375e-05
# # # # # 					iou filtering time: 0.002934694290161133
# # # # # 					stability score filtering time: 0.002923727035522461
# # # # # 					thresholding time: 0.0004222393035888672
# # # # # 					box filtering time: 8.821487426757812e-06
# # # # # 					mask uncrop time: 2.86102294921875e-06
# # # # # 					rle compression time: 2.6226043701171875e-06
# # # # # 				batch filtering time: 0.006372928619384766
# # # # # 			batch process time: 0.028904199600219727
# # # # # num iou preds before nms: torch.Size([47])
# # # # # 			batch nms time: 0.0005831718444824219
# # # # # num iou preds after nms: torch.Size([6])
# # # # # 			uncrop time: 0.0001010894775390625
# # # # # 		crop process time: 0.04000663757324219
# # # # # 		duplicate crop removal time: 0.0005536079406738281
# # # # # mask data segmentations len: 6
# # # # # 	mask generation time: 0.04060959815979004
# # # # # 	postprocess time: 2.384185791015625e-07
# # # # # 	rle encoding time: 5.4836273193359375e-06
# # # # # 	write MaskData: 7.772445678710938e-05
# # # # # number of bounding boxes: 11


# # # # # ~ extracting one mask ~
# # # # # num anns: 6
# # # # # img.shape: (720, 1280, 3)
# # # # # get best max: 1700424074.394832
# # # # # find intersection point: 2.384185791015625e-07
# # # # # set mask: 0.004079103469848633
# # # # # draw marker: 3.552436828613281e-05
# # # # # draw line mask + best bounding box: 4.1961669921875e-05

# # # # # encoder/decoder priming run: 0.5075891017913818
# # # # # all gaze engines priming run: 0.09439969062805176
# # # # # yolo priming run: 1.0870187282562256

# # # # # load img: 0.07396984100341797
# # # # # resize img: 1.6898672580718994
# # # # # generate masks: 0.04073834419250488
# # # # # detect face (primed): 0.0031282901763916016
# # # # # smooth + extract face (primed): 4.506111145019531e-05
# # # # # detect landmark (primed): 0.0008144378662109375
# # # # # smooth landmark (primed): 0.0005674362182617188
# # # # # detect gaze (primed): 0.003378629684448242
# # # # # smooth gaze (primed): 1.2636184692382812e-05
# # # # # visualize gaze: 0.0006630420684814453
# # # # # create plots: 5.4836273193359375e-06
# # # # # get gaze mask: 0.00025010108947753906
# # # # # prep yolo img: 0.0013720989227294922
# # # # # yolo pred: 0.0012717247009277344
# # # # # total yolo: 0.0026438236236572266
# # # # # draw and get yolo boxes: 0.00405120849609375
# # # # # segment one mask: 0.004708051681518555

# # # # # display image: 0.002256155014038086
# # # # # save to file (out/quantized_yolo/1700424079.6743982.png): 1.906562089920044
# # # # # non-load total: 0.061013221740722656
# # # # # load total: 0.9002969264984131
# # # # # """

# # # # res = """
# # # # python combo.py 

# # # # ~~~ ITER 1 with file ../base_imgs/gum.png ~~~
# # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # encoder preprocess time: 0.021246910095214844
# # # # prep encoder time: 1.0460848808288574
# # # # prep decoder time: 0.012819766998291016
# # # # iou access time: 2.384185791015625e-07
# # # # low res mask access time: 0.0
# # # # prep encoder time: 0.0017137527465820312
# # # # prep decoder time: 0.012150764465332031
# # # # iou access time: 2.384185791015625e-07
# # # # low res mask access time: 0.0
# # # # output shape: (2,)
# # # # Image Size: W=1280, H=720
# # # # output shape: (2,)
# # # # num crop boxes: 1
# # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # 			MASK ENCODER TIME: 0.012803792953491211
# # # # 			point preprocessing time: 4.76837158203125e-07
# # # # 				batch preprocess time: 0.004693269729614258
# # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # iou predictions shape: torch.Size([32, 4])
# # # # tensor([0.9341, 0.9359, 0.9943, 0.9963], device='cuda:0')
# # # # tensor([[ -8.3312,  -8.7316,  -8.2382,  ...,  -9.7016, -10.2523, -10.3068],
# # # #         [ -8.1383, -10.0043,  -8.2711,  ..., -13.8833, -17.8920, -10.7422],
# # # #         [ -8.1659,  -8.7316,  -8.5832,  ..., -10.3107, -13.8337, -10.2180],
# # # #         ...,
# # # #         [-11.0272, -18.2468, -12.3594,  ..., -15.0133, -12.1334, -12.9814],
# # # #         [-16.5785, -16.0865, -18.6799,  ..., -12.5112, -15.2204, -12.6444],
# # # #         [-11.6083, -17.3905, -11.8912,  ..., -14.5588, -12.3766, -12.4022]],
# # # #        device='cuda:0')
# # # # decoder running time: 0.012742757797241211
# # # # time to access iou predictions before postprocessing 0.010339498519897461
# # # # time to access low res masks before postprocessing 0.001558542251586914
# # # # 1.4066696166992188e-05
# # # # 4.291534423828125e-06
# # # # 6.890296936035156e-05
# # # # 3.0517578125e-05
# # # # 				BATCH DECODER TIME: 0.026261568069458008
# # # # done filtering iou
# # # # done filtering stability score
# # # # done filtering edges
# # # # 					convert to MaskData class: 0.0002181529998779297
# # # # 					keep mask access time: 5.626678466796875e-05
# # # # 					iou filtering time: 0.0021338462829589844
# # # # 					stability score filtering time: 0.0021708011627197266
# # # # 					thresholding time: 0.002285480499267578
# # # # 					box filtering time: 1.0013580322265625e-05
# # # # 					mask uncrop time: 3.5762786865234375e-06
# # # # 					rle compression time: 7.62939453125e-06
# # # # 				batch filtering time: 0.006829500198364258
# # # # 			batch process time: 0.03798961639404297
# # # # num iou preds before nms: torch.Size([29])
# # # # 			batch nms time: 0.0019898414611816406
# # # # num iou preds after nms: torch.Size([12])
# # # # 			uncrop time: 0.00016045570373535156
# # # # 		crop process time: 0.05372118949890137
# # # # 		duplicate crop removal time: 0.004580497741699219
# # # # mask data segmentations len: 12
# # # # 	mask generation time: 0.05836844444274902
# # # # 	postprocess time: 4.76837158203125e-07
# # # # 	rle encoding time: 8.106231689453125e-06
# # # # 	write MaskData: 0.00015282630920410156
# # # # number of bounding boxes: 18


# # # # ~ extracting one mask ~
# # # # num anns: 12
# # # # img.shape: (720, 1280, 3)
# # # # get best max: 1700424820.186949
# # # # find intersection point: 0.0
# # # # set mask: 0.0026175975799560547
# # # # draw marker: 5.1975250244140625e-05
# # # # draw line mask + best bounding box: 3.075599670410156e-05

# # # # encoder/decoder priming run: 1.594268560409546
# # # # all gaze engines priming run: 0.1013185977935791
# # # # yolo priming run: 1.0987391471862793

# # # # load img: 0.0655820369720459
# # # # resize img: 2.7952075004577637
# # # # generate masks: 0.058609724044799805
# # # # detect face (primed): 0.0027201175689697266
# # # # smooth + extract face (primed): 5.14984130859375e-05
# # # # detect landmark (primed): 0.0009367465972900391
# # # # smooth landmark (primed): 0.0006811618804931641
# # # # detect gaze (primed): 0.004065275192260742
# # # # smooth gaze (primed): 1.5974044799804688e-05
# # # # visualize gaze: 0.0008361339569091797
# # # # create plots: 8.344650268554688e-06
# # # # get gaze mask: 0.00037288665771484375
# # # # prep yolo img: 0.0031578540802001953
# # # # yolo pred: 0.0030128955841064453
# # # # total yolo: 0.006170749664306641
# # # # draw and get yolo boxes: 0.003050088882446289
# # # # segment one mask: 0.004158735275268555

# # # # display image: 0.019918203353881836
# # # # save to file (out/quantized_yolo/1700424822.7090611.png): 0.747307538986206
# # # # non-load total: 0.08168435096740723
# # # # load total: 10.538463354110718


# # # # ~~~ ITER 2 with file ../base_imgs/help.png ~~~
# # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # encoder preprocess time: 0.02386617660522461
# # # # prep encoder time: 0.004746913909912109
# # # # prep decoder time: 0.01224827766418457
# # # # iou access time: 2.384185791015625e-07
# # # # low res mask access time: 0.0
# # # # prep encoder time: 0.0017504692077636719
# # # # prep decoder time: 0.012148618698120117
# # # # iou access time: 4.76837158203125e-07
# # # # low res mask access time: 0.0
# # # # output shape: (2,)
# # # # Image Size: W=1280, H=720
# # # # output shape: (2,)
# # # # num crop boxes: 1
# # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # 			MASK ENCODER TIME: 0.012502431869506836
# # # # 			point preprocessing time: 9.5367431640625e-07
# # # # 				batch preprocess time: 0.0044918060302734375
# # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # iou predictions shape: torch.Size([32, 4])
# # # # tensor([0.9384, 0.9952, 0.9580, 0.9861], device='cuda:0')
# # # # tensor([[ -6.6941,  -7.8905,  -7.2229,  ...,  -9.0991,  -8.3330,  -8.6883],
# # # #         [ -6.4616, -11.2376,  -8.0175,  ..., -10.4587, -11.2644,  -8.8265],
# # # #         [ -6.5578,  -9.3998,  -7.6523,  ...,  -8.6371,  -9.2273,  -8.1202],
# # # #         ...,
# # # #         [-10.5942, -17.0922, -12.1792,  ..., -14.1470, -11.6094, -12.7005],
# # # #         [-14.8027, -14.9252, -16.9730,  ..., -11.9153, -13.6718, -11.9522],
# # # #         [-11.1681, -16.6521, -11.8045,  ..., -13.8949, -11.7971, -12.2474]],
# # # #        device='cuda:0')
# # # # decoder running time: 0.01292109489440918
# # # # time to access iou predictions before postprocessing 0.007001638412475586
# # # # time to access low res masks before postprocessing 0.0012655258178710938
# # # # 1.4066696166992188e-05
# # # # 4.291534423828125e-06
# # # # 3.719329833984375e-05
# # # # 1.7881393432617188e-05
# # # # 				BATCH DECODER TIME: 0.021581649780273438
# # # # done filtering iou
# # # # done filtering stability score
# # # # done filtering edges
# # # # 					convert to MaskData class: 9.703636169433594e-05
# # # # 					keep mask access time: 1.4543533325195312e-05
# # # # 					iou filtering time: 0.0028378963470458984
# # # # 					stability score filtering time: 0.0028553009033203125
# # # # 					thresholding time: 0.0020949840545654297
# # # # 					box filtering time: 8.821487426757812e-06
# # # # 					mask uncrop time: 4.291534423828125e-06
# # # # 					rle compression time: 7.62939453125e-06
# # # # 				batch filtering time: 0.007905960083007812
# # # # 			batch process time: 0.03411865234375
# # # # num iou preds before nms: torch.Size([55])
# # # # 			batch nms time: 0.0011219978332519531
# # # # num iou preds after nms: torch.Size([10])
# # # # 			uncrop time: 0.00011777877807617188
# # # # 		crop process time: 0.04856371879577637
# # # # 		duplicate crop removal time: 0.0009620189666748047
# # # # mask data segmentations len: 10
# # # # 	mask generation time: 0.049599409103393555
# # # # 	postprocess time: 7.152557373046875e-07
# # # # 	rle encoding time: 6.4373016357421875e-06
# # # # 	write MaskData: 0.00013709068298339844
# # # # number of bounding boxes: 10


# # # # ~ extracting one mask ~
# # # # num anns: 10
# # # # img.shape: (720, 1280, 3)
# # # # no box intersection
# # # # [   0. 6171.    0. 9594. 9663. 6168. 5917. 5892.    0. 1048.]
# # # # get best max: 1700424839.7180445
# # # # find intersection point: 2.384185791015625e-07
# # # # set mask: 0.0025768280029296875
# # # # draw marker: 5.435943603515625e-05
# # # # draw line mask + best bounding box: 6.198883056640625e-06

# # # # encoder/decoder priming run: 0.5715277194976807
# # # # all gaze engines priming run: 0.09804415702819824
# # # # yolo priming run: 1.0900802612304688

# # # # load img: 0.04367661476135254
# # # # resize img: 1.7607364654541016
# # # # generate masks: 0.049855947494506836
# # # # detect face (primed): 0.0033521652221679688
# # # # smooth + extract face (primed): 5.412101745605469e-05
# # # # detect landmark (primed): 0.0008778572082519531
# # # # smooth landmark (primed): 0.0005929470062255859
# # # # detect gaze (primed): 0.003851175308227539
# # # # smooth gaze (primed): 1.2874603271484375e-05
# # # # visualize gaze: 0.0007064342498779297
# # # # create plots: 6.4373016357421875e-06
# # # # get gaze mask: 0.0002758502960205078
# # # # prep yolo img: 0.001836538314819336
# # # # yolo pred: 0.002732992172241211
# # # # total yolo: 0.004569530487060547
# # # # draw and get yolo boxes: 0.0028913021087646484
# # # # segment one mask: 0.0039017200469970703

# # # # display image: 0.0045623779296875
# # # # save to file (out/quantized_yolo/1700424837.0115979.png): 0.9079999923706055
# # # # non-load total: 0.0709536075592041
# # # # load total: 0.8343007564544678


# # # # ~~~ ITER 3 with file ../base_imgs/pen.png ~~~
# # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # encoder preprocess time: 0.020885944366455078
# # # # prep encoder time: 0.004250288009643555
# # # # prep decoder time: 0.012279510498046875
# # # # iou access time: 2.384185791015625e-07
# # # # low res mask access time: 0.0
# # # # prep encoder time: 0.0017025470733642578
# # # # prep decoder time: 0.012040138244628906
# # # # iou access time: 2.384185791015625e-07
# # # # low res mask access time: 0.0
# # # # output shape: (2,)
# # # # Image Size: W=1280, H=720
# # # # output shape: (2,)
# # # # num crop boxes: 1
# # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # 			MASK ENCODER TIME: 0.012125253677368164
# # # # 			point preprocessing time: 7.152557373046875e-07
# # # # 				batch preprocess time: 0.004666328430175781
# # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # iou predictions shape: torch.Size([32, 4])
# # # # tensor([0.9368, 0.8405, 0.9989, 0.9847], device='cuda:0')
# # # # tensor([[ -7.7636,  -7.9784,  -7.5077,  ...,  -8.4606,  -8.7036,  -8.7648],
# # # #         [ -8.1390,  -9.4876,  -7.4140,  ...,  -9.8309, -11.5964,  -8.8396],
# # # #         [ -7.5226,  -7.9571,  -8.5773,  ...,  -7.9887,  -8.6826,  -7.9120],
# # # #         ...,
# # # #         [-11.2609, -18.5252, -12.6119,  ..., -14.5753, -11.7419, -13.0095],
# # # #         [-16.8256, -16.0771, -18.6756,  ..., -12.4279, -14.5041, -12.3878],
# # # #         [-12.2111, -17.9982, -12.4198,  ..., -14.4994, -11.9264, -12.6629]],
# # # #        device='cuda:0')
# # # # decoder running time: 0.012259483337402344
# # # # time to access iou predictions before postprocessing 0.006772041320800781
# # # # time to access low res masks before postprocessing 0.0011930465698242188
# # # # 1.7404556274414062e-05
# # # # 4.5299530029296875e-06
# # # # 2.3603439331054688e-05
# # # # 8.344650268554688e-06
# # # # 				BATCH DECODER TIME: 0.02057337760925293
# # # # done filtering iou
# # # # done filtering stability score
# # # # done filtering edges
# # # # 					convert to MaskData class: 8.606910705566406e-05
# # # # 					keep mask access time: 1.3828277587890625e-05
# # # # 					iou filtering time: 0.0027413368225097656
# # # # 					stability score filtering time: 0.0026481151580810547
# # # # 					thresholding time: 0.0004620552062988281
# # # # 					box filtering time: 8.344650268554688e-06
# # # # 					mask uncrop time: 2.6226043701171875e-06
# # # # 					rle compression time: 3.337860107421875e-06
# # # # 				batch filtering time: 0.005951881408691406
# # # # 			batch process time: 0.03125
# # # # num iou preds before nms: torch.Size([62])
# # # # 			batch nms time: 0.0006644725799560547
# # # # num iou preds after nms: torch.Size([11])
# # # # 			uncrop time: 0.00010704994201660156
# # # # 		crop process time: 0.04478883743286133
# # # # 		duplicate crop removal time: 0.0010352134704589844
# # # # mask data segmentations len: 11
# # # # 	mask generation time: 0.04600119590759277
# # # # 	postprocess time: 7.152557373046875e-07
# # # # 	rle encoding time: 6.67572021484375e-06
# # # # 	write MaskData: 0.00012946128845214844
# # # # number of bounding boxes: 18


# # # # ~ extracting one mask ~
# # # # num anns: 11
# # # # img.shape: (720, 1280, 3)
# # # # get best max: 1700424842.3673534
# # # # find intersection point: 2.384185791015625e-07
# # # # set mask: 0.005861759185791016
# # # # draw marker: 5.340576171875e-05
# # # # draw line mask + best bounding box: 2.2411346435546875e-05

# # # # encoder/decoder priming run: 0.5360395908355713
# # # # all gaze engines priming run: 0.09538149833679199
# # # # yolo priming run: 1.0812573432922363

# # # # load img: 0.07676577568054199
# # # # resize img: 1.713590383529663
# # # # generate masks: 0.046204328536987305
# # # # detect face (primed): 0.002178668975830078
# # # # smooth + extract face (primed): 4.601478576660156e-05
# # # # detect landmark (primed): 0.000835418701171875
# # # # smooth landmark (primed): 0.0005600452423095703
# # # # detect gaze (primed): 0.003592252731323242
# # # # smooth gaze (primed): 1.3113021850585938e-05
# # # # visualize gaze: 0.0006563663482666016
# # # # create plots: 6.9141387939453125e-06
# # # # get gaze mask: 0.00035572052001953125
# # # # prep yolo img: 0.001463174819946289
# # # # yolo pred: 0.0025582313537597656
# # # # total yolo: 0.004021406173706055
# # # # draw and get yolo boxes: 0.0030755996704101562
# # # # segment one mask: 0.007333517074584961

# # # # display image: 0.002738475799560547
# # # # save to file (out/quantized_yolo/1700424840.7237027.png): 1.1862118244171143
# # # # non-load total: 0.06888461112976074
# # # # load total: 0.7910115718841553


# # # # ~~~ ITER 4 with file ../base_imgs/psycho.png ~~~
# # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # encoder preprocess time: 0.02261042594909668
# # # # prep encoder time: 0.00422978401184082
# # # # prep decoder time: 0.012347221374511719
# # # # iou access time: 2.384185791015625e-07
# # # # low res mask access time: 2.384185791015625e-07
# # # # prep encoder time: 0.001745462417602539
# # # # prep decoder time: 0.011995315551757812
# # # # iou access time: 0.0
# # # # low res mask access time: 2.384185791015625e-07
# # # # output shape: (2,)
# # # # Image Size: W=1280, H=720
# # # # output shape: (2,)
# # # # num crop boxes: 1
# # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # 			MASK ENCODER TIME: 0.012079000473022461
# # # # 			point preprocessing time: 7.152557373046875e-07
# # # # 				batch preprocess time: 0.004349470138549805
# # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # iou predictions shape: torch.Size([32, 4])
# # # # tensor([0.9256, 0.9281, 0.9981, 0.9906], device='cuda:0')
# # # # tensor([[ -8.1298,  -8.7053,  -8.1575,  ...,  -7.5086,  -8.6267,  -8.3698],
# # # #         [ -7.6153, -11.9814,  -9.2905,  ...,  -9.1115, -10.3391,  -9.4338],
# # # #         [ -7.3439,  -9.2675,  -8.3370,  ...,  -8.3399,  -8.8131,  -8.9123],
# # # #         ...,
# # # #         [-11.3914, -17.6585, -12.5132,  ..., -13.4607, -11.2967, -12.2024],
# # # #         [-16.8266, -16.2217, -18.1999,  ..., -12.0473, -13.3000, -11.7340],
# # # #         [-12.3472, -17.6093, -12.3978,  ..., -13.7099, -11.4338, -12.0383]],
# # # #        device='cuda:0')
# # # # decoder running time: 0.011810779571533203
# # # # time to access iou predictions before postprocessing 0.006738424301147461
# # # # time to access low res masks before postprocessing 0.0011966228485107422
# # # # 1.3113021850585938e-05
# # # # 4.291534423828125e-06
# # # # 2.765655517578125e-05
# # # # 1.2874603271484375e-05
# # # # 				BATCH DECODER TIME: 0.020084857940673828
# # # # done filtering iou
# # # # done filtering stability score
# # # # done filtering edges
# # # # 					convert to MaskData class: 7.677078247070312e-05
# # # # 					keep mask access time: 1.2636184692382812e-05
# # # # 					iou filtering time: 0.0027594566345214844
# # # # 					stability score filtering time: 0.002677440643310547
# # # # 					thresholding time: 0.0004620552062988281
# # # # 					box filtering time: 8.58306884765625e-06
# # # # 					mask uncrop time: 4.76837158203125e-06
# # # # 					rle compression time: 3.337860107421875e-06
# # # # 				batch filtering time: 0.005992412567138672
# # # # 			batch process time: 0.030484914779663086
# # # # num iou preds before nms: torch.Size([62])
# # # # 			batch nms time: 0.0006692409515380859
# # # # num iou preds after nms: torch.Size([13])
# # # # 			uncrop time: 0.00010824203491210938
# # # # 		crop process time: 0.044005393981933594
# # # # 		duplicate crop removal time: 0.004149436950683594
# # # # mask data segmentations len: 13
# # # # 	mask generation time: 0.048217058181762695
# # # # 	postprocess time: 2.384185791015625e-07
# # # # 	rle encoding time: 5.9604644775390625e-06
# # # # 	write MaskData: 0.00014162063598632812
# # # # number of bounding boxes: 13


# # # # ~ extracting one mask ~
# # # # num anns: 13
# # # # img.shape: (720, 1280, 3)
# # # # get best max: 1700424847.2875807
# # # # find intersection point: 0.0
# # # # set mask: 0.0059452056884765625
# # # # draw marker: 4.863739013671875e-05
# # # # draw line mask + best bounding box: 2.6464462280273438e-05

# # # # encoder/decoder priming run: 0.540313720703125
# # # # all gaze engines priming run: 0.09546232223510742
# # # # yolo priming run: 1.0736801624298096

# # # # load img: 0.07734251022338867
# # # # resize img: 1.7108688354492188
# # # # generate masks: 0.04844379425048828
# # # # detect face (primed): 0.002190828323364258
# # # # smooth + extract face (primed): 5.650520324707031e-05
# # # # detect landmark (primed): 0.0008494853973388672
# # # # smooth landmark (primed): 0.0005707740783691406
# # # # detect gaze (primed): 0.0035758018493652344
# # # # smooth gaze (primed): 1.3113021850585938e-05
# # # # visualize gaze: 0.0006401538848876953
# # # # create plots: 6.67572021484375e-06
# # # # get gaze mask: 0.0004036426544189453
# # # # prep yolo img: 0.0017642974853515625
# # # # yolo pred: 0.0025186538696289062
# # # # total yolo: 0.004282951354980469
# # # # draw and get yolo boxes: 0.002964496612548828
# # # # segment one mask: 0.007436513900756836

# # # # display image: 0.0029256343841552734
# # # # save to file (out/quantized_yolo/1700424844.6227362.png): 1.3638741970062256
# # # # non-load total: 0.07143974304199219
# # # # load total: 0.811854362487793


# # # # ~~~ ITER 5 with file ../base_imgs/workpls_v2.png ~~~
# # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # encoder preprocess time: 0.0217592716217041
# # # # prep encoder time: 0.0040836334228515625
# # # # prep decoder time: 0.012330293655395508
# # # # iou access time: 0.0
# # # # low res mask access time: 2.384185791015625e-07
# # # # prep encoder time: 0.0017132759094238281
# # # # prep decoder time: 0.012000083923339844
# # # # iou access time: 2.384185791015625e-07
# # # # low res mask access time: 0.0
# # # # output shape: (2,)
# # # # Image Size: W=1280, H=720
# # # # output shape: (2,)
# # # # num crop boxes: 1
# # # # 			crop preprocess time: 1.9073486328125e-06
# # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # 			MASK ENCODER TIME: 0.012897014617919922
# # # # 			point preprocessing time: 7.152557373046875e-07
# # # # 				batch preprocess time: 0.004054069519042969
# # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # iou predictions shape: torch.Size([32, 4])
# # # # tensor([0.8279, 0.9650, 0.9918, 0.9998], device='cuda:0')
# # # # tensor([[ -6.9110,  -7.1324,  -6.6747,  ...,  -7.2847,  -7.7170,  -7.4182],
# # # #         [ -6.6451,  -7.8450,  -6.5016,  ...,  -8.9482,  -9.6664,  -9.2111],
# # # #         [ -7.2869,  -7.6486,  -7.3117,  ...,  -7.3160,  -7.7708,  -7.9341],
# # # #         ...,
# # # #         [ -9.2495, -14.0668, -10.2365,  ..., -10.5705,  -9.3030,  -9.3800],
# # # #         [-13.2842, -12.9731, -14.6998,  ...,  -9.4204, -10.4889,  -9.0765],
# # # #         [ -9.8340, -13.9412, -10.0812,  ..., -10.8513,  -9.4465,  -9.3463]],
# # # #        device='cuda:0')
# # # # decoder running time: 0.011780261993408203
# # # # time to access iou predictions before postprocessing 0.006722211837768555
# # # # time to access low res masks before postprocessing 0.0011844635009765625
# # # # 1.3589859008789062e-05
# # # # 4.291534423828125e-06
# # # # 2.47955322265625e-05
# # # # 1.1444091796875e-05
# # # # 				BATCH DECODER TIME: 0.020020484924316406
# # # # done filtering iou
# # # # done filtering stability score
# # # # done filtering edges
# # # # 					convert to MaskData class: 0.00010704994201660156
# # # # 					keep mask access time: 1.2636184692382812e-05
# # # # 					iou filtering time: 0.003050088882446289
# # # # 					stability score filtering time: 0.0036573410034179688
# # # # 					thresholding time: 0.0004253387451171875
# # # # 					box filtering time: 7.867813110351562e-06
# # # # 					mask uncrop time: 5.0067901611328125e-06
# # # # 					rle compression time: 3.0994415283203125e-06
# # # # 				batch filtering time: 0.0072557926177978516
# # # # 			batch process time: 0.031389474868774414
# # # # num iou preds before nms: torch.Size([78])
# # # # 			batch nms time: 0.0006072521209716797
# # # # num iou preds after nms: torch.Size([8])
# # # # 			uncrop time: 0.00010251998901367188
# # # # 		crop process time: 0.04566669464111328
# # # # 		duplicate crop removal time: 0.0008130073547363281
# # # # mask data segmentations len: 8
# # # # 	mask generation time: 0.046540260314941406
# # # # 	postprocess time: 4.76837158203125e-07
# # # # 	rle encoding time: 5.245208740234375e-06
# # # # 	write MaskData: 0.00010395050048828125
# # # # number of bounding boxes: 2


# # # # ~ extracting one mask ~
# # # # num anns: 8
# # # # img.shape: (720, 1280, 3)
# # # # get best max: 1700424851.414621
# # # # find intersection point: 2.384185791015625e-07
# # # # set mask: 0.0024254322052001953
# # # # draw marker: 3.9577484130859375e-05
# # # # draw line mask + best bounding box: 1.9788742065429688e-05

# # # # encoder/decoder priming run: 0.5387601852416992
# # # # all gaze engines priming run: 0.09663033485412598
# # # # yolo priming run: 1.0757825374603271

# # # # load img: 0.08280444145202637
# # # # resize img: 1.7121257781982422
# # # # generate masks: 0.04672646522521973
# # # # detect face (primed): 0.002359628677368164
# # # # smooth + extract face (primed): 6.4849853515625e-05
# # # # detect landmark (primed): 0.0008783340454101562
# # # # smooth landmark (primed): 0.0006039142608642578
# # # # detect gaze (primed): 0.0036988258361816406
# # # # smooth gaze (primed): 1.4781951904296875e-05
# # # # visualize gaze: 0.0006413459777832031
# # # # create plots: 5.9604644775390625e-06
# # # # get gaze mask: 0.0002942085266113281
# # # # prep yolo img: 0.0014796257019042969
# # # # yolo pred: 0.002476930618286133
# # # # total yolo: 0.00395655632019043
# # # # draw and get yolo boxes: 0.002569437026977539
# # # # segment one mask: 0.0030944347381591797

# # # # display image: 0.003268718719482422
# # # # save to file (out/quantized_yolo/1700424848.7202475.png): 1.759920358657837
# # # # non-load total: 0.06491518020629883
# # # # load total: 0.8376331329345703


# # # # ~~~ ITER 6 with file ../base_imgs/zz.png ~~~
# # # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # # encoder preprocess time: 0.019906044006347656
# # # # prep encoder time: 0.00404667854309082
# # # # prep decoder time: 0.012345552444458008
# # # # iou access time: 2.384185791015625e-07
# # # # low res mask access time: 2.384185791015625e-07
# # # # prep encoder time: 0.0017673969268798828
# # # # prep decoder time: 0.012015104293823242
# # # # iou access time: 0.0
# # # # low res mask access time: 2.384185791015625e-07
# # # # output shape: (2,)
# # # # Image Size: W=1280, H=720
# # # # output shape: (2,)
# # # # num crop boxes: 1
# # # # 			crop preprocess time: 1.6689300537109375e-06
# # # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # # 			MASK ENCODER TIME: 0.012614965438842773
# # # # 			point preprocessing time: 9.5367431640625e-07
# # # # 				batch preprocess time: 0.0039997100830078125
# # # # mask shape: torch.Size([32, 4, 256, 256])
# # # # iou predictions shape: torch.Size([32, 4])
# # # # tensor([0.9005, 0.9935, 0.9940, 0.9868], device='cuda:0')
# # # # tensor([[ -7.2592,  -7.6928,  -7.8967,  ...,  -9.1549,  -8.8214,  -9.7043],
# # # #         [ -6.2857,  -9.0861,  -7.5322,  ..., -10.4792, -10.5100, -11.1403],
# # # #         [ -6.9098,  -8.4500,  -8.0675,  ...,  -8.3816,  -8.5970,  -9.1251],
# # # #         ...,
# # # #         [-10.5805, -17.5918, -12.0115,  ..., -14.7354, -12.2225, -12.9182],
# # # #         [-15.3859, -15.0354, -17.6378,  ..., -12.0299, -13.8576, -12.0151],
# # # #         [-11.2866, -17.0498, -11.9494,  ..., -14.7120, -12.5366, -12.6470]],
# # # #        device='cuda:0')
# # # # decoder running time: 0.011829853057861328
# # # # time to access iou predictions before postprocessing 0.006726741790771484
# # # # time to access low res masks before postprocessing 0.0011715888977050781
# # # # 1.33514404296875e-05
# # # # 4.76837158203125e-06
# # # # 1.7404556274414062e-05
# # # # 7.867813110351562e-06
# # # # 				BATCH DECODER TIME: 0.020037412643432617
# # # # done filtering iou
# # # # done filtering stability score
# # # # done filtering edges
# # # # 					convert to MaskData class: 7.128715515136719e-05
# # # # 					keep mask access time: 1.1205673217773438e-05
# # # # 					iou filtering time: 0.002881288528442383
# # # # 					stability score filtering time: 0.0028839111328125
# # # # 					thresholding time: 0.0004382133483886719
# # # # 					box filtering time: 8.58306884765625e-06
# # # # 					mask uncrop time: 2.1457672119140625e-06
# # # # 					rle compression time: 2.86102294921875e-06
# # # # 				batch filtering time: 0.006288290023803711
# # # # 			batch process time: 0.030376911163330078
# # # # num iou preds before nms: torch.Size([48])
# # # # 			batch nms time: 0.0005750656127929688
# # # # num iou preds after nms: torch.Size([7])
# # # # 			uncrop time: 0.00010275840759277344
# # # # 		crop process time: 0.04428577423095703
# # # # 		duplicate crop removal time: 0.0013642311096191406
# # # # mask data segmentations len: 7
# # # # 	mask generation time: 0.0457000732421875
# # # # 	postprocess time: 4.76837158203125e-07
# # # # 	rle encoding time: 5.245208740234375e-06
# # # # 	write MaskData: 7.867813110351562e-05
# # # # number of bounding boxes: 11


# # # # ~ extracting one mask ~
# # # # num anns: 7
# # # # img.shape: (720, 1280, 3)
# # # # get best max: 1700424847.835474
# # # # find intersection point: 2.384185791015625e-07
# # # # set mask: 0.0040242671966552734
# # # # draw marker: 3.552436828613281e-05
# # # # draw line mask + best bounding box: 4.38690185546875e-05

# # # # encoder/decoder priming run: 0.5346760749816895
# # # # all gaze engines priming run: 0.09603309631347656
# # # # yolo priming run: 1.0695316791534424

# # # # load img: 0.0728445053100586
# # # # resize img: 1.7016489505767822
# # # # generate masks: 0.04583382606506348
# # # # detect face (primed): 0.0022699832916259766
# # # # smooth + extract face (primed): 4.363059997558594e-05
# # # # detect landmark (primed): 0.0008597373962402344
# # # # smooth landmark (primed): 0.0006062984466552734
# # # # detect gaze (primed): 0.003710508346557617
# # # # smooth gaze (primed): 1.33514404296875e-05
# # # # visualize gaze: 0.0006146430969238281
# # # # create plots: 5.9604644775390625e-06
# # # # get gaze mask: 0.0002779960632324219
# # # # prep yolo img: 0.0015759468078613281
# # # # yolo pred: 0.0027494430541992188
# # # # total yolo: 0.004325389862060547
# # # # draw and get yolo boxes: 0.002553701400756836
# # # # segment one mask: 0.0048062801361083984

# # # # display image: 0.0020971298217773438
# # # # save to file (out/quantized_yolo/1700424853.237888.png): 1.9071879386901855
# # # # non-load total: 0.06592679023742676
# # # # load total: 0.761894941329956

# # # # (efficientvit) nicole@k9:~/gaze_sam/integration$ 
# # # # """

# # # res = """

# # # ~~~ ITER 1 with file ../base_imgs/gum.png ~~~
# # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # encoder preprocess time: 0.028209686279296875
# # # prep encoder time: 0.014661550521850586
# # # prep decoder time: 0.0013384819030761719
# # # iou access time: 2.384185791015625e-07
# # # low res mask access time: 2.384185791015625e-07
# # # prep encoder time: 0.001226186752319336
# # # prep decoder time: 0.001405954360961914
# # # iou access time: 2.384185791015625e-07
# # # low res mask access time: 2.384185791015625e-07
# # # output shape: (2,)
# # # Image Size: W=1280, H=720
# # # output shape: (2,)
# # # num crop boxes: 1
# # # 			crop preprocess time: 2.1457672119140625e-06
# # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # 			MASK ENCODER TIME: 0.012363672256469727
# # # 			point preprocessing time: 7.152557373046875e-07
# # # 				batch preprocess time: 0.01110696792602539
# # # mask shape: torch.Size([32, 4, 256, 256])
# # # iou predictions shape: torch.Size([32, 4])
# # # tensor([0.9332, 0.9382, 0.9951, 0.9988], device='cuda:0')
# # # tensor([[ -8.6754,  -9.1052,  -8.5505,  ..., -10.1330, -10.6225, -10.8266],
# # #         [ -8.5585, -10.5391,  -8.6980,  ..., -14.8327, -19.3889, -11.2556],
# # #         [ -8.5686,  -9.1190,  -8.9597,  ..., -10.8657, -14.7500, -10.7599],
# # #         ...,
# # #         [-11.5922, -19.3524, -13.1172,  ..., -15.9189, -12.8250, -13.6923],
# # #         [-17.5557, -17.0089, -19.8860,  ..., -13.1702, -16.0622, -13.2976],
# # #         [-12.2838, -18.4534, -12.6782,  ..., -15.4346, -13.1001, -13.0277]],
# # #        device='cuda:0')
# # # decoder running time: 0.0009698867797851562
# # # time to access iou predictions before postprocessing 0.061019182205200195
# # # time to access low res masks before postprocessing 0.0018579959869384766
# # # 1.9550323486328125e-05
# # # 6.198883056640625e-06
# # # 5.7220458984375e-05
# # # 1.621246337890625e-05
# # # 				BATCH DECODER TIME: 0.06685304641723633
# # # done filtering iou
# # # done filtering stability score
# # # done filtering edges
# # # 					convert to MaskData class: 0.00019550323486328125
# # # 					keep mask access time: 6.222724914550781e-05
# # # 					iou filtering time: 0.005691051483154297
# # # 					stability score filtering time: 0.004579782485961914
# # # 					thresholding time: 0.007523298263549805
# # # 					box filtering time: 1.3589859008789062e-05
# # # 					mask uncrop time: 3.5762786865234375e-06
# # # 					rle compression time: 3.814697265625e-06
# # # 				batch filtering time: 0.018010616302490234
# # # 			batch process time: 0.09603404998779297
# # # num iou preds before nms: torch.Size([27])
# # # 			batch nms time: 0.001580953598022461
# # # num iou preds after nms: torch.Size([12])
# # # 			uncrop time: 0.00018548965454101562
# # # 		crop process time: 0.11113882064819336
# # # 		duplicate crop removal time: 0.010766744613647461
# # # mask data segmentations len: 12
# # # 	mask generation time: 0.12197542190551758
# # # 	postprocess time: 1.1920928955078125e-06
# # # 	rle encoding time: 8.106231689453125e-06
# # # 	write MaskData: 0.00015115737915039062
# # # number of bounding boxes: 18


# # # ~ extracting one mask ~
# # # num anns: 12
# # # img.shape: (720, 1280, 3)
# # # get best max: 1700430253.2115786
# # # find intersection point: 4.76837158203125e-07
# # # set mask: 0.0022580623626708984
# # # draw marker: 5.3882598876953125e-05
# # # draw line mask + best bounding box: 2.2649765014648438e-05

# # # encoder/decoder priming run: 0.9577462673187256
# # # all gaze engines priming run: 0.5172982215881348
# # # yolo priming run: 1.4872806072235107

# # # load img: 0.09263944625854492
# # # resize img: 2.9647655487060547
# # # generate masks: 0.12221908569335938
# # # detect face (primed): 0.0025131702423095703
# # # smooth + extract face (primed): 7.510185241699219e-05
# # # detect landmark (primed): 0.0012047290802001953
# # # smooth landmark (primed): 0.0006184577941894531
# # # detect gaze (primed): 0.004171848297119141
# # # smooth gaze (primed): 1.7881393432617188e-05
# # # visualize gaze: 0.0007476806640625
# # # create plots: 7.62939453125e-06
# # # get gaze mask: 0.0004642009735107422
# # # prep yolo img: 0.004117727279663086
# # # yolo pred: 0.0022063255310058594
# # # total yolo: 0.006324052810668945
# # # draw and get yolo boxes: 0.00582575798034668
# # # segment one mask: 0.004224538803100586

# # # display image: 0.04161882400512695
# # # save to file (out/quantized_yolo/1700430259.7536724.png): 0.9236776828765869
# # # non-load total: 0.1484208106994629
# # # load total: 6.255731821060181


# # # ~~~ ITER 2 with file ../base_imgs/help.png ~~~
# # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # encoder preprocess time: 0.02745223045349121
# # # prep encoder time: 0.0016872882843017578
# # # prep decoder time: 0.0008661746978759766
# # # iou access time: 2.384185791015625e-07
# # # low res mask access time: 2.384185791015625e-07
# # # prep encoder time: 0.001230478286743164
# # # prep decoder time: 0.0006966590881347656
# # # iou access time: 2.384185791015625e-07
# # # low res mask access time: 0.0
# # # output shape: (2,)
# # # Image Size: W=1280, H=720
# # # output shape: (2,)
# # # num crop boxes: 1
# # # 			crop preprocess time: 2.384185791015625e-06
# # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # 			MASK ENCODER TIME: 0.012177467346191406
# # # 			point preprocessing time: 7.152557373046875e-07
# # # 				batch preprocess time: 0.01194453239440918
# # # mask shape: torch.Size([32, 4, 256, 256])
# # # iou predictions shape: torch.Size([32, 4])
# # # tensor([0.9354, 0.9963, 0.9622, 0.9898], device='cuda:0')
# # # tensor([[ -7.0355,  -8.3566,  -7.6345,  ...,  -9.8546,  -8.9164,  -9.3916],
# # #         [ -6.8851, -12.2454,  -8.6850,  ..., -11.4576, -12.4912,  -9.5266],
# # #         [ -6.9913, -10.1939,  -8.2552,  ...,  -9.3990, -10.0679,  -8.7645],
# # #         ...,
# # #         [-11.3972, -18.5436, -13.2509,  ..., -15.3709, -12.5928, -13.7247],
# # #         [-15.9674, -16.1429, -18.4776,  ..., -12.8494, -14.7534, -12.8752],
# # #         [-12.0868, -18.0809, -12.9045,  ..., -15.0994, -12.8118, -13.1785]],
# # #        device='cuda:0')
# # # decoder running time: 0.0009694099426269531
# # # time to access iou predictions before postprocessing 0.0643768310546875
# # # time to access low res masks before postprocessing 0.0018131732940673828
# # # 2.1457672119140625e-05
# # # 5.4836273193359375e-06
# # # 1.9788742065429688e-05
# # # 8.106231689453125e-06
# # # 				BATCH DECODER TIME: 0.06764817237854004
# # # done filtering iou
# # # done filtering stability score
# # # done filtering edges
# # # 					convert to MaskData class: 0.000102996826171875
# # # 					keep mask access time: 1.7881393432617188e-05
# # # 					iou filtering time: 0.007217884063720703
# # # 					stability score filtering time: 0.007000446319580078
# # # 					thresholding time: 0.0020253658294677734
# # # 					box filtering time: 1.1444091796875e-05
# # # 					mask uncrop time: 3.0994415283203125e-06
# # # 					rle compression time: 3.5762786865234375e-06
# # # 				batch filtering time: 0.01636481285095215
# # # 			batch process time: 0.09601736068725586
# # # num iou preds before nms: torch.Size([59])
# # # 			batch nms time: 0.0009870529174804688
# # # num iou preds after nms: torch.Size([7])
# # # 			uncrop time: 0.0001552104949951172
# # # 		crop process time: 0.11025166511535645
# # # 		duplicate crop removal time: 0.0016894340515136719
# # # mask data segmentations len: 7
# # # 	mask generation time: 0.11200833320617676
# # # 	postprocess time: 7.152557373046875e-07
# # # 	rle encoding time: 8.344650268554688e-06
# # # 	write MaskData: 0.00011205673217773438
# # # number of bounding boxes: 10


# # # ~ extracting one mask ~
# # # num anns: 7
# # # img.shape: (720, 1280, 3)
# # # no box intersection
# # # [6178.    0.    0. 9634. 6190. 5919. 1043.]
# # # get best max: 1700430274.125472
# # # find intersection point: 4.76837158203125e-07
# # # set mask: 0.002299070358276367
# # # draw marker: 5.4836273193359375e-05
# # # draw line mask + best bounding box: 7.3909759521484375e-06

# # # encoder/decoder priming run: 0.9151322841644287
# # # all gaze engines priming run: 0.13170433044433594
# # # yolo priming run: 1.4559612274169922

# # # load img: 0.050112247467041016
# # # resize img: 2.5047597885131836
# # # generate masks: 0.11219048500061035
# # # detect face (primed): 0.0061550140380859375
# # # smooth + extract face (primed): 7.557868957519531e-05
# # # detect landmark (primed): 0.0012640953063964844
# # # smooth landmark (primed): 0.0006139278411865234
# # # detect gaze (primed): 0.004271268844604492
# # # smooth gaze (primed): 2.002716064453125e-05
# # # visualize gaze: 0.0007045269012451172
# # # create plots: 7.152557373046875e-06
# # # get gaze mask: 0.00022721290588378906
# # # prep yolo img: 0.0038399696350097656
# # # yolo pred: 0.0014081001281738281
# # # total yolo: 0.005248069763183594
# # # draw and get yolo boxes: 0.006718873977661133
# # # segment one mask: 0.003590822219848633

# # # display image: 0.0035796165466308594
# # # save to file (out/quantized_yolo/1700430270.1956377.png): 1.061340570449829
# # # non-load total: 0.14109396934509277
# # # load total: 1.2368199825286865


# # # ~~~ ITER 3 with file ../base_imgs/pen.png ~~~
# # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # encoder preprocess time: 0.02729654312133789
# # # prep encoder time: 0.0016629695892333984
# # # prep decoder time: 0.0008578300476074219
# # # iou access time: 2.384185791015625e-07
# # # low res mask access time: 4.76837158203125e-07
# # # prep encoder time: 0.0012257099151611328
# # # prep decoder time: 0.0007109642028808594
# # # iou access time: 0.0
# # # low res mask access time: 2.384185791015625e-07
# # # output shape: (2,)
# # # Image Size: W=1280, H=720
# # # output shape: (2,)
# # # num crop boxes: 1
# # # 			crop preprocess time: 2.1457672119140625e-06
# # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # 			MASK ENCODER TIME: 0.012782812118530273
# # # 			point preprocessing time: 4.76837158203125e-07
# # # 				batch preprocess time: 0.010619640350341797
# # # mask shape: torch.Size([32, 4, 256, 256])
# # # iou predictions shape: torch.Size([32, 4])
# # # tensor([0.9374, 0.8518, 1.0003, 0.9865], device='cuda:0')
# # # tensor([[ -8.0734,  -8.3174,  -7.8178,  ...,  -8.7720,  -9.0142,  -9.1708],
# # #         [ -8.5753, -10.0236,  -7.7946,  ..., -10.2595, -12.2916,  -9.1482],
# # #         [ -7.9301,  -8.3592,  -9.0283,  ...,  -8.3091,  -9.0415,  -8.2489],
# # #         ...,
# # #         [-11.7714, -19.5810, -13.2925,  ..., -15.3810, -12.3283, -13.6815],
# # #         [-17.7426, -16.9203, -19.8171,  ..., -13.0544, -15.2874, -13.0053],
# # #         [-12.8166, -19.0180, -13.1442,  ..., -15.3029, -12.5349, -13.2805]],
# # #        device='cuda:0')
# # # decoder running time: 0.0017261505126953125
# # # time to access iou predictions before postprocessing 0.05782580375671387
# # # time to access low res masks before postprocessing 0.0018305778503417969
# # # 2.2172927856445312e-05
# # # 5.9604644775390625e-06
# # # 2.5987625122070312e-05
# # # 8.821487426757812e-06
# # # 				BATCH DECODER TIME: 0.0619049072265625
# # # done filtering iou
# # # done filtering stability score
# # # done filtering edges
# # # 					convert to MaskData class: 0.00011038780212402344
# # # 					keep mask access time: 2.002716064453125e-05
# # # 					iou filtering time: 0.006665706634521484
# # # 					stability score filtering time: 0.006523609161376953
# # # 					thresholding time: 0.0006582736968994141
# # # 					box filtering time: 1.239776611328125e-05
# # # 					mask uncrop time: 3.0994415283203125e-06
# # # 					rle compression time: 3.337860107421875e-06
# # # 				batch filtering time: 0.013976812362670898
# # # 			batch process time: 0.0865776538848877
# # # num iou preds before nms: torch.Size([68])
# # # 			batch nms time: 0.001420736312866211
# # # num iou preds after nms: torch.Size([15])
# # # 			uncrop time: 0.0001766681671142578
# # # 		crop process time: 0.1018824577331543
# # # 		duplicate crop removal time: 0.003301858901977539
# # # mask data segmentations len: 15
# # # 	mask generation time: 0.10526323318481445
# # # 	postprocess time: 1.1920928955078125e-06
# # # 	rle encoding time: 8.58306884765625e-06
# # # 	write MaskData: 0.0001926422119140625
# # # number of bounding boxes: 18


# # # ~ extracting one mask ~
# # # num anns: 15
# # # img.shape: (720, 1280, 3)
# # # get best max: 1700430278.137274
# # # find intersection point: 7.152557373046875e-07
# # # set mask: 0.005430936813354492
# # # draw marker: 5.53131103515625e-05
# # # draw line mask + best bounding box: 2.4557113647460938e-05

# # # encoder/decoder priming run: 0.8633317947387695
# # # all gaze engines priming run: 0.12108230590820312
# # # yolo priming run: 1.4416124820709229

# # # load img: 0.09803128242492676
# # # resize img: 2.428333044052124
# # # generate masks: 0.10554695129394531
# # # detect face (primed): 0.006215095520019531
# # # smooth + extract face (primed): 8.273124694824219e-05
# # # detect landmark (primed): 0.0016252994537353516
# # # smooth landmark (primed): 0.0005986690521240234
# # # detect gaze (primed): 0.005023956298828125
# # # smooth gaze (primed): 1.8358230590820312e-05
# # # visualize gaze: 0.0006968975067138672
# # # create plots: 7.62939453125e-06
# # # get gaze mask: 0.0004863739013671875
# # # prep yolo img: 0.0036301612854003906
# # # yolo pred: 0.0014090538024902344
# # # total yolo: 0.005039215087890625
# # # draw and get yolo boxes: 0.006167411804199219
# # # segment one mask: 0.007976770401000977

# # # display image: 0.0039517879486083984
# # # save to file (out/quantized_yolo/1700430275.2493525.png): 1.3141379356384277
# # # non-load total: 0.13949131965637207
# # # load total: 1.2301111221313477


# # # ~~~ ITER 4 with file ../base_imgs/psycho.png ~~~
# # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # encoder preprocess time: 0.027474403381347656
# # # prep encoder time: 0.001636505126953125
# # # prep decoder time: 0.0008795261383056641
# # # iou access time: 4.76837158203125e-07
# # # low res mask access time: 0.0
# # # prep encoder time: 0.0012590885162353516
# # # prep decoder time: 0.000705718994140625
# # # iou access time: 2.384185791015625e-07
# # # low res mask access time: 0.0
# # # output shape: (2,)
# # # Image Size: W=1280, H=720
# # # output shape: (2,)
# # # num crop boxes: 1
# # # 			crop preprocess time: 2.384185791015625e-06
# # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # 			MASK ENCODER TIME: 0.013267040252685547
# # # 			point preprocessing time: 7.152557373046875e-07
# # # 				batch preprocess time: 0.010579586029052734
# # # mask shape: torch.Size([32, 4, 256, 256])
# # # iou predictions shape: torch.Size([32, 4])
# # # tensor([0.9296, 0.9290, 1.0003, 0.9924], device='cuda:0')
# # # tensor([[ -8.5027,  -9.1174,  -8.4902,  ...,  -7.9094,  -8.9963,  -8.8265],
# # #         [ -8.0348, -12.9851,  -9.9456,  ...,  -9.6382, -11.1637,  -9.9886],
# # #         [ -7.7692,  -9.8898,  -8.8434,  ...,  -8.8937,  -9.2840,  -9.5097],
# # #         ...,
# # #         [-11.9645, -18.8679, -13.2922,  ..., -14.3369, -11.9520, -12.9147],
# # #         [-17.8875, -17.2256, -19.5287,  ..., -12.7414, -14.1121, -12.4084],
# # #         [-13.0512, -18.7947, -13.2444,  ..., -14.5878, -12.0983, -12.6797]],
# # #        device='cuda:0')
# # # decoder running time: 0.0010612010955810547
# # # time to access iou predictions before postprocessing 0.05823802947998047
# # # time to access low res masks before postprocessing 0.0019087791442871094
# # # 2.3603439331054688e-05
# # # 7.152557373046875e-06
# # # 2.2649765014648438e-05
# # # 7.867813110351562e-06
# # # 				BATCH DECODER TIME: 0.061730384826660156
# # # done filtering iou
# # # done filtering stability score
# # # done filtering edges
# # # 					convert to MaskData class: 0.00010585784912109375
# # # 					keep mask access time: 1.7881393432617188e-05
# # # 					iou filtering time: 0.006524324417114258
# # # 					stability score filtering time: 0.0059967041015625
# # # 					thresholding time: 0.000598907470703125
# # # 					box filtering time: 1.3113021850585938e-05
# # # 					mask uncrop time: 3.0994415283203125e-06
# # # 					rle compression time: 3.5762786865234375e-06
# # # 				batch filtering time: 0.013245582580566406
# # # 			batch process time: 0.0856330394744873
# # # num iou preds before nms: torch.Size([64])
# # # 			batch nms time: 0.00133514404296875
# # # num iou preds after nms: torch.Size([12])
# # # 			uncrop time: 0.00020551681518554688
# # # 		crop process time: 0.10133838653564453
# # # 		duplicate crop removal time: 0.0027174949645996094
# # # mask data segmentations len: 12
# # # 	mask generation time: 0.10412478446960449
# # # 	postprocess time: 1.1920928955078125e-06
# # # 	rle encoding time: 8.106231689453125e-06
# # # 	write MaskData: 0.00016045570373535156
# # # number of bounding boxes: 13


# # # ~ extracting one mask ~
# # # num anns: 12
# # # img.shape: (720, 1280, 3)
# # # get best max: 1700430284.6348546
# # # find intersection point: 2.384185791015625e-07
# # # set mask: 0.005575656890869141
# # # draw marker: 4.9114227294921875e-05
# # # draw line mask + best bounding box: 2.8371810913085938e-05

# # # encoder/decoder priming run: 0.8796138763427734
# # # all gaze engines priming run: 0.12151908874511719
# # # yolo priming run: 1.4874944686889648

# # # load img: 0.10445141792297363
# # # resize img: 2.490877151489258
# # # generate masks: 0.1043555736541748
# # # detect face (primed): 0.00561070442199707
# # # smooth + extract face (primed): 9.107589721679688e-05
# # # detect landmark (primed): 0.0013301372528076172
# # # smooth landmark (primed): 0.0006105899810791016
# # # detect gaze (primed): 0.004417896270751953
# # # smooth gaze (primed): 1.7404556274414062e-05
# # # visualize gaze: 0.0006742477416992188
# # # create plots: 7.152557373046875e-06
# # # get gaze mask: 0.00046133995056152344
# # # prep yolo img: 0.003886699676513672
# # # yolo pred: 0.0014324188232421875
# # # total yolo: 0.005319118499755859
# # # draw and get yolo boxes: 0.006061077117919922
# # # segment one mask: 0.007476091384887695

# # # display image: 0.0036847591400146484
# # # save to file (out/quantized_yolo/1700430280.5643466.png): 1.5683951377868652
# # # non-load total: 0.13643932342529297
# # # load total: 1.3451893329620361


# # # ~~~ ITER 5 with file ../base_imgs/workpls_v2.png ~~~
# # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # encoder preprocess time: 0.02748727798461914
# # # prep encoder time: 0.001628875732421875
# # # prep decoder time: 0.0009009838104248047
# # # iou access time: 4.76837158203125e-07
# # # low res mask access time: 0.0
# # # prep encoder time: 0.0012555122375488281
# # # prep decoder time: 0.0007076263427734375
# # # iou access time: 2.384185791015625e-07
# # # low res mask access time: 0.0
# # # output shape: (2,)
# # # Image Size: W=1280, H=720
# # # output shape: (2,)
# # # num crop boxes: 1
# # # 			crop preprocess time: 2.1457672119140625e-06
# # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # 			MASK ENCODER TIME: 0.012701749801635742
# # # 			point preprocessing time: 4.76837158203125e-07
# # # 				batch preprocess time: 0.0106964111328125
# # # mask shape: torch.Size([32, 4, 256, 256])
# # # iou predictions shape: torch.Size([32, 4])
# # # tensor([0.8421, 0.9651, 0.9942, 1.0011], device='cuda:0')
# # # tensor([[ -7.3482,  -7.6141,  -7.0780,  ...,  -7.7793,  -8.1702,  -7.9282],
# # #         [ -7.1271,  -8.4585,  -6.9817,  ...,  -9.7545, -10.5299, -10.0313],
# # #         [ -7.8270,  -8.2133,  -7.8469,  ...,  -7.8565,  -8.3059,  -8.5654],
# # #         ...,
# # #         [ -9.9201, -15.2971, -11.1023,  ..., -11.4544, -10.0333, -10.1079],
# # #         [-14.3791, -14.0383, -16.0287,  ..., -10.1411, -11.3077,  -9.7647],
# # #         [-10.6079, -15.1685, -10.9687,  ..., -11.7700, -10.1843, -10.0488]],
# # #        device='cuda:0')
# # # decoder running time: 0.0009808540344238281
# # # time to access iou predictions before postprocessing 0.058580875396728516
# # # time to access low res masks before postprocessing 0.0018470287322998047
# # # 2.0503997802734375e-05
# # # 6.4373016357421875e-06
# # # 2.09808349609375e-05
# # # 7.152557373046875e-06
# # # 				BATCH DECODER TIME: 0.06191420555114746
# # # done filtering iou
# # # done filtering stability score
# # # done filtering edges
# # # 					convert to MaskData class: 0.00010704994201660156
# # # 					keep mask access time: 1.7642974853515625e-05
# # # 					iou filtering time: 0.007319211959838867
# # # 					stability score filtering time: 0.008851766586303711
# # # 					thresholding time: 0.0006456375122070312
# # # 					box filtering time: 1.2159347534179688e-05
# # # 					mask uncrop time: 2.86102294921875e-06
# # # 					rle compression time: 3.337860107421875e-06
# # # 				batch filtering time: 0.01694202423095703
# # # 			batch process time: 0.08962821960449219
# # # num iou preds before nms: torch.Size([101])
# # # 			batch nms time: 0.002100229263305664
# # # num iou preds after nms: torch.Size([8])
# # # 			uncrop time: 0.00017118453979492188
# # # 		crop process time: 0.10550832748413086
# # # 		duplicate crop removal time: 0.0018978118896484375
# # # mask data segmentations len: 8
# # # 	mask generation time: 0.10747480392456055
# # # 	postprocess time: 7.152557373046875e-07
# # # 	rle encoding time: 8.106231689453125e-06
# # # 	write MaskData: 0.00012159347534179688
# # # number of bounding boxes: 2


# # # ~ extracting one mask ~
# # # num anns: 8
# # # img.shape: (720, 1280, 3)
# # # get best max: 1700430290.183219
# # # find intersection point: 2.384185791015625e-07
# # # set mask: 0.002176523208618164
# # # draw marker: 4.601478576660156e-05
# # # draw line mask + best bounding box: 3.3855438232421875e-05

# # # encoder/decoder priming run: 0.8687429428100586
# # # all gaze engines priming run: 0.12186455726623535
# # # yolo priming run: 1.4570116996765137

# # # load img: 0.1018056869506836
# # # resize img: 2.4498062133789062
# # # generate masks: 0.1076667308807373
# # # detect face (primed): 0.005372524261474609
# # # smooth + extract face (primed): 7.581710815429688e-05
# # # detect landmark (primed): 0.0012483596801757812
# # # smooth landmark (primed): 0.0006170272827148438
# # # detect gaze (primed): 0.0042018890380859375
# # # smooth gaze (primed): 1.8835067749023438e-05
# # # visualize gaze: 0.0006897449493408203
# # # create plots: 8.344650268554688e-06
# # # get gaze mask: 0.0003597736358642578
# # # prep yolo img: 0.0038614273071289062
# # # yolo pred: 0.0014171600341796875
# # # total yolo: 0.005278587341308594
# # # draw and get yolo boxes: 0.006127595901489258
# # # segment one mask: 0.003125905990600586

# # # display image: 0.0036706924438476562
# # # save to file (out/quantized_yolo/1700430286.2702994.png): 2.057447910308838
# # # non-load total: 0.1347975730895996
# # # load total: 1.2309706211090088


# # # ~~~ ITER 6 with file ../base_imgs/zz.png ~~~
# # # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # # encoder preprocess time: 0.02733135223388672
# # # prep encoder time: 0.0016357898712158203
# # # prep decoder time: 0.0008866786956787109
# # # iou access time: 2.384185791015625e-07
# # # low res mask access time: 2.384185791015625e-07
# # # prep encoder time: 0.0012617111206054688
# # # prep decoder time: 0.0007121562957763672
# # # iou access time: 0.0
# # # low res mask access time: 2.384185791015625e-07
# # # output shape: (2,)
# # # Image Size: W=1280, H=720
# # # output shape: (2,)
# # # num crop boxes: 1
# # # 			crop preprocess time: 2.384185791015625e-06
# # # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # # 			MASK ENCODER TIME: 0.012444257736206055
# # # 			point preprocessing time: 7.152557373046875e-07
# # # 				batch preprocess time: 0.010666131973266602
# # # mask shape: torch.Size([32, 4, 256, 256])
# # # iou predictions shape: torch.Size([32, 4])
# # # tensor([0.9004, 0.9947, 0.9967, 0.9903], device='cuda:0')
# # # tensor([[ -7.6506,  -8.1205,  -8.3735,  ...,  -9.8069,  -9.3667, -10.3981],
# # #         [ -6.6363,  -9.6891,  -8.0045,  ..., -11.3278, -11.3665, -12.0270],
# # #         [ -7.3352,  -9.0430,  -8.6132,  ...,  -9.0217,  -9.1871,  -9.8192],
# # #         ...,
# # #         [-11.3059, -19.0020, -12.9464,  ..., -15.8896, -13.1473, -13.8592],
# # #         [-16.5402, -16.1738, -19.1463,  ..., -12.9348, -14.9116, -12.9168],
# # #         [-12.1219, -18.4048, -12.9387,  ..., -15.9090, -13.5248, -13.5571]],
# # #        device='cuda:0')
# # # decoder running time: 0.0010356903076171875
# # # time to access iou predictions before postprocessing 0.058261871337890625
# # # time to access low res masks before postprocessing 0.0017986297607421875
# # # 2.0503997802734375e-05
# # # 5.9604644775390625e-06
# # # 2.0742416381835938e-05
# # # 7.3909759521484375e-06
# # # 				BATCH DECODER TIME: 0.06160449981689453
# # # done filtering iou
# # # done filtering stability score
# # # done filtering edges
# # # 					convert to MaskData class: 0.00010228157043457031
# # # 					keep mask access time: 1.6927719116210938e-05
# # # 					iou filtering time: 0.0069353580474853516
# # # 					stability score filtering time: 0.006755352020263672
# # # 					thresholding time: 0.0006248950958251953
# # # 					box filtering time: 1.2159347534179688e-05
# # # 					mask uncrop time: 3.0994415283203125e-06
# # # 					rle compression time: 3.0994415283203125e-06
# # # 				batch filtering time: 0.01443624496459961
# # # 			batch process time: 0.08678221702575684
# # # num iou preds before nms: torch.Size([48])
# # # 			batch nms time: 0.0008716583251953125
# # # num iou preds after nms: torch.Size([7])
# # # 			uncrop time: 0.000152587890625
# # # 		crop process time: 0.10111284255981445
# # # 		duplicate crop removal time: 0.0017015933990478516
# # # mask data segmentations len: 7
# # # 	mask generation time: 0.10288047790527344
# # # 	postprocess time: 7.152557373046875e-07
# # # 	rle encoding time: 8.58306884765625e-06
# # # 	write MaskData: 0.00011038780212402344
# # # number of bounding boxes: 11


# # # ~ extracting one mask ~
# # # num anns: 7
# # # img.shape: (720, 1280, 3)
# # # get best max: 1700430288.198552
# # # find intersection point: 2.384185791015625e-07
# # # set mask: 0.0036933422088623047
# # # draw marker: 4.696846008300781e-05
# # # draw line mask + best bounding box: 4.506111145019531e-05

# # # encoder/decoder priming run: 0.9039902687072754
# # # all gaze engines priming run: 0.1246185302734375
# # # yolo priming run: 1.4382472038269043

# # # load img: 0.09387898445129395
# # # resize img: 2.469005823135376
# # # generate masks: 0.10306072235107422
# # # detect face (primed): 0.005593538284301758
# # # smooth + extract face (primed): 7.367134094238281e-05
# # # detect landmark (primed): 0.001226186752319336
# # # smooth landmark (primed): 0.0005903244018554688
# # # detect gaze (primed): 0.004198551177978516
# # # smooth gaze (primed): 1.8835067749023438e-05
# # # visualize gaze: 0.0006952285766601562
# # # create plots: 9.059906005859375e-06
# # # get gaze mask: 0.0003485679626464844
# # # prep yolo img: 0.003704071044921875
# # # yolo pred: 0.0014083385467529297
# # # total yolo: 0.005112409591674805
# # # draw and get yolo boxes: 0.006072521209716797
# # # segment one mask: 0.0047206878662109375

# # # display image: 0.003593921661376953
# # # save to file (out/quantized_yolo/1700430292.3032382.png): 2.158421754837036
# # # non-load total: 0.1317272186279297
# # # load total: 1.2052326202392578
# # # """

# # res = """

# # ~~~ ITER 1 with file ../base_imgs/gum.png ~~~
# # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # encoder preprocess time: 0.021251678466796875
# # prep encoder time: 0.08459973335266113
# # prep decoder time: 0.0014743804931640625
# # iou access time: 4.76837158203125e-07
# # low res mask access time: 0.0
# # prep encoder time: 0.0013043880462646484
# # prep decoder time: 0.0006990432739257812
# # iou access time: 0.0
# # low res mask access time: 2.384185791015625e-07
# # output shape: (2,)
# # Image Size: W=1280, H=720
# # output shape: (2,)
# # num crop boxes: 1
# # 			crop preprocess time: 1.6689300537109375e-06
# # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # 			MASK ENCODER TIME: 0.012194633483886719
# # 			point preprocessing time: 4.76837158203125e-07
# # 				batch preprocess time: 0.004201412200927734
# # 				BATCH DECODER TIME: 0.006944894790649414
# # done filtering iou
# # done filtering stability score
# # done filtering edges
# # 					convert to MaskData class: 0.0019898414611816406
# # 					keep mask access time: 0.0006833076477050781
# # 					iou filtering time: 0.017589807510375977
# # 					stability score filtering time: 0.0024483203887939453
# # 					thresholding time: 0.0063512325286865234
# # 					box filtering time: 1.2159347534179688e-05
# # 					mask uncrop time: 9.5367431640625e-06
# # 					rle compression time: 3.814697265625e-06
# # 				batch filtering time: 0.028404712677001953
# # 			batch process time: 0.03966689109802246
# # num iou preds before nms: torch.Size([29])
# # 			batch nms time: 0.0062503814697265625
# # num iou preds after nms: torch.Size([12])
# # 			uncrop time: 0.0002353191375732422
# # 		crop process time: 0.06042981147766113
# # 		duplicate crop removal time: 0.005705356597900391
# # mask data segmentations len: 12
# # 	mask generation time: 0.06624698638916016
# # 	postprocess time: 7.152557373046875e-07
# # 	rle encoding time: 8.821487426757812e-06
# # 	write MaskData: 0.0002295970916748047
# # number of bounding boxes: 18


# # ~ extracting one mask ~
# # num anns: 12
# # img.shape: (720, 1280, 3)
# # get best max: 1700682720.8367577
# # find intersection point: 2.384185791015625e-07
# # set mask: 0.0026149749755859375
# # draw marker: 4.982948303222656e-05
# # draw line mask + best bounding box: 3.0517578125e-05

# # encoder/decoder priming run: 0.5892355442047119
# # all gaze engines priming run: 0.1577746868133545
# # yolo priming run: 1.1454439163208008

# # load img: 0.06669139862060547
# # resize img: 1.893902063369751
# # generate masks: 0.06664586067199707
# # detect face (primed): 0.005776405334472656
# # smooth + extract face (primed): 0.00010561943054199219
# # detect landmark (primed): 0.0015180110931396484
# # smooth landmark (primed): 0.0007395744323730469
# # detect gaze (primed): 0.006500720977783203
# # smooth gaze (primed): 1.71661376953125e-05
# # visualize gaze: 0.0016965866088867188
# # create plots: 1.4543533325195312e-05
# # get gaze mask: 0.0004875659942626953
# # prep yolo img: 0.0021593570709228516
# # yolo pred: 0.002215147018432617
# # total yolo: 0.004374504089355469
# # draw and get yolo boxes: 0.004196882247924805
# # segment one mask: 0.004204750061035156

# # display image: 0.0416109561920166
# # save to file (out/quantized_yolo/1700682731.8087697.png): 0.7410380840301514
# # non-load total: 0.09628701210021973
# # load total: 2.974219560623169


# # ~~~ ITER 2 with file ../base_imgs/help.png ~~~
# # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # encoder preprocess time: 0.020874738693237305
# # prep encoder time: 0.001981973648071289
# # prep decoder time: 0.0008230209350585938
# # iou access time: 2.384185791015625e-07
# # low res mask access time: 2.384185791015625e-07
# # prep encoder time: 0.0009431838989257812
# # prep decoder time: 0.0004918575286865234
# # iou access time: 0.0
# # low res mask access time: 2.384185791015625e-07
# # output shape: (2,)
# # Image Size: W=1280, H=720
# # output shape: (2,)
# # num crop boxes: 1
# # 			crop preprocess time: 1.430511474609375e-06
# # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # 			MASK ENCODER TIME: 0.010024070739746094
# # 			point preprocessing time: 2.384185791015625e-07
# # 				batch preprocess time: 0.004434823989868164
# # 				BATCH DECODER TIME: 0.0014190673828125
# # done filtering iou
# # done filtering stability score
# # done filtering edges
# # 					convert to MaskData class: 0.0001049041748046875
# # 					keep mask access time: 1.2636184692382812e-05
# # 					iou filtering time: 0.017800569534301758
# # 					stability score filtering time: 0.002928018569946289
# # 					thresholding time: 0.0017974376678466797
# # 					box filtering time: 1.0251998901367188e-05
# # 					mask uncrop time: 2.6226043701171875e-06
# # 					rle compression time: 2.6226043701171875e-06
# # 				batch filtering time: 0.022646427154541016
# # 			batch process time: 0.028560876846313477
# # num iou preds before nms: torch.Size([60])
# # 			batch nms time: 0.0009675025939941406
# # num iou preds after nms: torch.Size([6])
# # 			uncrop time: 0.000118255615234375
# # 		crop process time: 0.04035305976867676
# # 		duplicate crop removal time: 0.0010123252868652344
# # mask data segmentations len: 6
# # 	mask generation time: 0.0414125919342041
# # 	postprocess time: 7.152557373046875e-07
# # 	rle encoding time: 5.9604644775390625e-06
# # 	write MaskData: 7.05718994140625e-05
# # number of bounding boxes: 10


# # ~ extracting one mask ~
# # num anns: 6
# # img.shape: (720, 1280, 3)
# # no box intersection
# # [   0. 6179.    0. 9621. 5891.  987.]
# # get best max: 1700682740.205395
# # find intersection point: 2.384185791015625e-07
# # set mask: 0.002575397491455078
# # draw marker: 4.458427429199219e-05
# # draw line mask + best bounding box: 7.867813110351562e-06

# # encoder/decoder priming run: 0.47693943977355957
# # all gaze engines priming run: 0.09532594680786133
# # yolo priming run: 1.087618112564087

# # load img: 0.04161810874938965
# # resize img: 1.6606342792510986
# # generate masks: 0.04154253005981445
# # detect face (primed): 0.0028238296508789062
# # smooth + extract face (primed): 5.340576171875e-05
# # detect landmark (primed): 0.0008568763732910156
# # smooth landmark (primed): 0.0005562305450439453
# # detect gaze (primed): 0.003908634185791016
# # smooth gaze (primed): 1.1682510375976562e-05
# # visualize gaze: 0.0007228851318359375
# # create plots: 6.198883056640625e-06
# # get gaze mask: 0.0001652240753173828
# # prep yolo img: 0.002527952194213867
# # yolo pred: 0.0014066696166992188
# # total yolo: 0.003934621810913086
# # draw and get yolo boxes: 0.0037927627563476562
# # segment one mask: 0.0036194324493408203

# # display image: 0.0023097991943359375
# # save to file (out/quantized_yolo/1700682737.630736.png): 0.8828125
# # non-load total: 0.06200051307678223
# # load total: 0.8139944076538086


# # ~~~ ITER 3 with file ../base_imgs/pen.png ~~~
# # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # encoder preprocess time: 0.020624876022338867
# # prep encoder time: 0.00180816650390625
# # prep decoder time: 0.0007824897766113281
# # iou access time: 0.0
# # low res mask access time: 2.384185791015625e-07
# # prep encoder time: 0.0009353160858154297
# # prep decoder time: 0.0004801750183105469
# # iou access time: 0.0
# # low res mask access time: 0.0
# # output shape: (2,)
# # Image Size: W=1280, H=720
# # output shape: (2,)
# # num crop boxes: 1
# # 			crop preprocess time: 1.1920928955078125e-06
# # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # 			MASK ENCODER TIME: 0.009595394134521484
# # 			point preprocessing time: 1.1920928955078125e-06
# # 				batch preprocess time: 0.00445866584777832
# # 				BATCH DECODER TIME: 0.0012557506561279297
# # done filtering iou
# # done filtering stability score
# # done filtering edges
# # 					convert to MaskData class: 9.870529174804688e-05
# # 					keep mask access time: 1.33514404296875e-05
# # 					iou filtering time: 0.017901897430419922
# # 					stability score filtering time: 0.0028672218322753906
# # 					thresholding time: 0.0005218982696533203
# # 					box filtering time: 8.58306884765625e-06
# # 					mask uncrop time: 2.6226043701171875e-06
# # 					rle compression time: 2.6226043701171875e-06
# # 				batch filtering time: 0.02140355110168457
# # 			batch process time: 0.027179479598999023
# # num iou preds before nms: torch.Size([65])
# # 			batch nms time: 0.0007181167602539062
# # num iou preds after nms: torch.Size([12])
# # 			uncrop time: 0.000118255615234375
# # 		crop process time: 0.03828692436218262
# # 		duplicate crop removal time: 0.0017752647399902344
# # mask data segmentations len: 12
# # 	mask generation time: 0.04011845588684082
# # 	postprocess time: 4.76837158203125e-07
# # 	rle encoding time: 5.9604644775390625e-06
# # 	write MaskData: 0.00014328956604003906
# # number of bounding boxes: 18


# # ~ extracting one mask ~
# # num anns: 12
# # img.shape: (720, 1280, 3)
# # get best max: 1700682742.805117
# # find intersection point: 2.384185791015625e-07
# # set mask: 0.006810426712036133
# # draw marker: 5.316734313964844e-05
# # draw line mask + best bounding box: 2.8133392333984375e-05

# # encoder/decoder priming run: 0.5199160575866699
# # all gaze engines priming run: 0.09500575065612793
# # yolo priming run: 1.0733401775360107

# # load img: 0.0794229507446289
# # resize img: 1.6893806457519531
# # generate masks: 0.040318965911865234
# # detect face (primed): 0.0034515857696533203
# # smooth + extract face (primed): 4.124641418457031e-05
# # detect landmark (primed): 0.0007963180541992188
# # smooth landmark (primed): 0.0005385875701904297
# # detect gaze (primed): 0.0034804344177246094
# # smooth gaze (primed): 1.1682510375976562e-05
# # visualize gaze: 0.0006964206695556641
# # create plots: 6.4373016357421875e-06
# # get gaze mask: 0.00034737586975097656
# # prep yolo img: 0.0016167163848876953
# # yolo pred: 0.0012767314910888672
# # total yolo: 0.0028934478759765625
# # draw and get yolo boxes: 0.004015922546386719
# # segment one mask: 0.008476972579956055

# # display image: 0.002453327178955078
# # save to file (out/quantized_yolo/1700682741.12983.png): 1.1226844787597656
# # non-load total: 0.06508064270019531
# # load total: 0.8489751815795898


# # ~~~ ITER 4 with file ../base_imgs/psycho.png ~~~
# # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # encoder preprocess time: 0.023056507110595703
# # prep encoder time: 0.0018725395202636719
# # prep decoder time: 0.0007903575897216797
# # iou access time: 2.384185791015625e-07
# # low res mask access time: 2.384185791015625e-07
# # prep encoder time: 0.0009412765502929688
# # prep decoder time: 0.0004851818084716797
# # iou access time: 2.384185791015625e-07
# # low res mask access time: 0.0
# # output shape: (2,)
# # Image Size: W=1280, H=720
# # output shape: (2,)
# # num crop boxes: 1
# # 			crop preprocess time: 1.430511474609375e-06
# # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # 			MASK ENCODER TIME: 0.00990915298461914
# # 			point preprocessing time: 1.1920928955078125e-06
# # 				batch preprocess time: 0.004399299621582031
# # 				BATCH DECODER TIME: 0.001482248306274414
# # done filtering iou
# # done filtering stability score
# # done filtering edges
# # 					convert to MaskData class: 9.751319885253906e-05
# # 					keep mask access time: 1.3828277587890625e-05
# # 					iou filtering time: 0.017700672149658203
# # 					stability score filtering time: 0.0027620792388916016
# # 					thresholding time: 0.0005123615264892578
# # 					box filtering time: 8.821487426757812e-06
# # 					mask uncrop time: 3.5762786865234375e-06
# # 					rle compression time: 2.384185791015625e-06
# # 				batch filtering time: 0.0210874080657959
# # 			batch process time: 0.027030467987060547
# # num iou preds before nms: torch.Size([69])
# # 			batch nms time: 0.0007085800170898438
# # num iou preds after nms: torch.Size([13])
# # 			uncrop time: 0.00011134147644042969
# # 		crop process time: 0.03846478462219238
# # 		duplicate crop removal time: 0.0019271373748779297
# # mask data segmentations len: 13
# # 	mask generation time: 0.04045391082763672
# # 	postprocess time: 7.152557373046875e-07
# # 	rle encoding time: 6.198883056640625e-06
# # 	write MaskData: 0.00019121170043945312
# # number of bounding boxes: 13


# # ~ extracting one mask ~
# # num anns: 13
# # img.shape: (720, 1280, 3)
# # get best max: 1700682747.5538335
# # find intersection point: 2.384185791015625e-07
# # set mask: 0.006907463073730469
# # draw marker: 6.437301635742188e-05
# # draw line mask + best bounding box: 3.147125244140625e-05

# # encoder/decoder priming run: 0.47890353202819824
# # all gaze engines priming run: 0.09470558166503906
# # yolo priming run: 1.0765342712402344

# # load img: 0.07766532897949219
# # resize img: 1.651489019393921
# # generate masks: 0.040726661682128906
# # detect face (primed): 0.003785848617553711
# # smooth + extract face (primed): 5.3882598876953125e-05
# # detect landmark (primed): 0.0009510517120361328
# # smooth landmark (primed): 0.0005548000335693359
# # detect gaze (primed): 0.0037467479705810547
# # smooth gaze (primed): 1.2159347534179688e-05
# # visualize gaze: 0.0007174015045166016
# # create plots: 5.245208740234375e-06
# # get gaze mask: 0.0003616809844970703
# # prep yolo img: 0.0015206336975097656
# # yolo pred: 0.0013260841369628906
# # total yolo: 0.0028467178344726562
# # draw and get yolo boxes: 0.003908872604370117
# # segment one mask: 0.008507251739501953

# # display image: 0.002942800521850586
# # save to file (out/quantized_yolo/1700682744.9698727.png): 1.351754903793335
# # non-load total: 0.06618499755859375
# # load total: 0.7961981296539307


# # ~~~ ITER 5 with file ../base_imgs/workpls_v2.png ~~~
# # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # encoder preprocess time: 0.020629405975341797
# # prep encoder time: 0.001714944839477539
# # prep decoder time: 0.0008168220520019531
# # iou access time: 2.384185791015625e-07
# # low res mask access time: 0.0
# # prep encoder time: 0.0009543895721435547
# # prep decoder time: 0.0004832744598388672
# # iou access time: 0.0
# # low res mask access time: 2.384185791015625e-07
# # output shape: (2,)
# # Image Size: W=1280, H=720
# # output shape: (2,)
# # num crop boxes: 1
# # 			crop preprocess time: 1.6689300537109375e-06
# # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # 			MASK ENCODER TIME: 0.009660005569458008
# # 			point preprocessing time: 4.76837158203125e-07
# # 				batch preprocess time: 0.004495143890380859
# # 				BATCH DECODER TIME: 0.0012352466583251953
# # done filtering iou
# # done filtering stability score
# # done filtering edges
# # 					convert to MaskData class: 9.059906005859375e-05
# # 					keep mask access time: 1.3113021850585938e-05
# # 					iou filtering time: 0.018153667449951172
# # 					stability score filtering time: 0.0038750171661376953
# # 					thresholding time: 0.0005056858062744141
# # 					box filtering time: 8.106231689453125e-06
# # 					mask uncrop time: 2.6226043701171875e-06
# # 					rle compression time: 2.384185791015625e-06
# # 				batch filtering time: 0.02263808250427246
# # 			batch process time: 0.028424978256225586
# # num iou preds before nms: torch.Size([101])
# # 			batch nms time: 0.0008699893951416016
# # num iou preds after nms: torch.Size([8])
# # 			uncrop time: 0.000110626220703125
# # 		crop process time: 0.03973793983459473
# # 		duplicate crop removal time: 0.001249074935913086
# # mask data segmentations len: 8
# # 	mask generation time: 0.04103565216064453
# # 	postprocess time: 4.76837158203125e-07
# # 	rle encoding time: 6.4373016357421875e-06
# # 	write MaskData: 8.654594421386719e-05
# # number of bounding boxes: 2


# # ~ extracting one mask ~
# # num anns: 8
# # img.shape: (720, 1280, 3)
# # get best max: 1700682751.6300118
# # find intersection point: 0.0
# # set mask: 0.002490997314453125
# # draw marker: 3.457069396972656e-05
# # draw line mask + best bounding box: 2.6702880859375e-05

# # encoder/decoder priming run: 0.5092315673828125
# # all gaze engines priming run: 0.0957193374633789
# # yolo priming run: 1.0729663372039795

# # load img: 0.0818018913269043
# # resize img: 1.6791744232177734
# # generate masks: 0.04117417335510254
# # detect face (primed): 0.0030562877655029297
# # smooth + extract face (primed): 4.100799560546875e-05
# # detect landmark (primed): 0.0008091926574707031
# # smooth landmark (primed): 0.0006072521209716797
# # detect gaze (primed): 0.003368854522705078
# # smooth gaze (primed): 1.239776611328125e-05
# # visualize gaze: 0.0007114410400390625
# # create plots: 5.245208740234375e-06
# # get gaze mask: 0.0002772808074951172
# # prep yolo img: 0.001527547836303711
# # yolo pred: 0.0012841224670410156
# # total yolo: 0.0028116703033447266
# # draw and get yolo boxes: 0.003825664520263672
# # segment one mask: 0.00315093994140625

# # display image: 0.0022192001342773438
# # save to file (out/quantized_yolo/1700682748.9485261.png): 1.747856616973877
# # non-load total: 0.05985617637634277
# # load total: 0.8638415336608887


# # ~~~ ITER 6 with file ../base_imgs/zz.png ~~~
# # loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# # encoder preprocess time: 0.020489931106567383
# # prep encoder time: 0.001748800277709961
# # prep decoder time: 0.000812530517578125
# # iou access time: 2.384185791015625e-07
# # low res mask access time: 0.0
# # prep encoder time: 0.00095367431640625
# # prep decoder time: 0.0004773139953613281
# # iou access time: 0.0
# # low res mask access time: 0.0
# # output shape: (2,)
# # Image Size: W=1280, H=720
# # output shape: (2,)
# # num crop boxes: 1
# # 			crop preprocess time: 1.430511474609375e-06
# # image shape after preprocess: torch.Size([1, 3, 512, 512])
# # features after passing through encoder: torch.Size([1, 256, 64, 64])
# # 			MASK ENCODER TIME: 0.00975346565246582
# # 			point preprocessing time: 4.76837158203125e-07
# # 				batch preprocess time: 0.00444483757019043
# # 				BATCH DECODER TIME: 0.001483917236328125
# # done filtering iou
# # done filtering stability score
# # done filtering edges
# # 					convert to MaskData class: 9.846687316894531e-05
# # 					keep mask access time: 1.3113021850585938e-05
# # 					iou filtering time: 0.01783895492553711
# # 					stability score filtering time: 0.002939939498901367
# # 					thresholding time: 0.0004966259002685547
# # 					box filtering time: 8.58306884765625e-06
# # 					mask uncrop time: 2.1457672119140625e-06
# # 					rle compression time: 2.86102294921875e-06
# # 				batch filtering time: 0.021387577056884766
# # 			batch process time: 0.027409076690673828
# # num iou preds before nms: torch.Size([47])
# # 			batch nms time: 0.0006847381591796875
# # num iou preds after nms: torch.Size([6])
# # 			uncrop time: 0.00011205673217773438
# # 		crop process time: 0.03862714767456055
# # 		duplicate crop removal time: 0.0009298324584960938
# # mask data segmentations len: 6
# # 	mask generation time: 0.0396120548248291
# # 	postprocess time: 4.76837158203125e-07
# # 	rle encoding time: 5.9604644775390625e-06
# # 	write MaskData: 8.344650268554688e-05
# # number of bounding boxes: 11


# # ~ extracting one mask ~
# # num anns: 6
# # img.shape: (720, 1280, 3)
# # get best max: 1700682747.98588
# # find intersection point: 2.384185791015625e-07
# # set mask: 0.004270315170288086
# # draw marker: 3.647804260253906e-05
# # draw line mask + best bounding box: 4.696846008300781e-05

# # encoder/decoder priming run: 0.4760096073150635
# # all gaze engines priming run: 0.09448623657226562
# # yolo priming run: 1.07700777053833

# # load img: 0.07569503784179688
# # resize img: 1.6487727165222168
# # generate masks: 0.03975987434387207
# # detect face (primed): 0.0035092830657958984
# # smooth + extract face (primed): 5.269050598144531e-05
# # detect landmark (primed): 0.0009763240814208984
# # smooth landmark (primed): 0.0005502700805664062
# # detect gaze (primed): 0.0036618709564208984
# # smooth gaze (primed): 1.1920928955078125e-05
# # visualize gaze: 0.0007715225219726562
# # create plots: 7.152557373046875e-06
# # get gaze mask: 0.00027871131896972656
# # prep yolo img: 0.0014846324920654297
# # yolo pred: 0.0012693405151367188
# # total yolo: 0.0027539730072021484
# # draw and get yolo boxes: 0.003987312316894531
# # segment one mask: 0.004908084869384766

# # display image: 0.0028028488159179688
# # save to file (out/quantized_yolo/1700682753.4138916.png): 1.8838789463043213
# # non-load total: 0.061235666275024414
# # load total: 0.7911972999572754

# # (efficientvit) nicole@k9:~/gaze_sam/integration$ 

# # """

# res = """
# (efficientvit) nicole@k9:~$ cd gaze_sam/integration/
# (efficientvit) nicole@k9:~/gaze_sam/integration$ bash build_fp16_decoder.sh 
# [11/22/2023-22:17:43] [TRT] [I] [MemUsageChange] Init CUDA: CPU +385, GPU +0, now: CPU 497, GPU 325 (MiB)
# [11/22/2023-22:17:49] [TRT] [I] [MemUsageChange] Init builder kernel library: CPU +1441, GPU +266, now: CPU 2014, GPU 591 (MiB)
# [11/22/2023-22:17:49] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
# /NFS/home/nicole/gaze_sam/integration/build_fp16_decoder.py:62: DeprecationWarning: Use set_memory_pool_limit instead.
#   self.config.max_workspace_size = 100 * (2 ** 30)  # old was 8 GB, but said it wasn't enough memory for all tactics
# network object: <tensorrt.tensorrt.INetworkDefinition object at 0x7f23060723f0>
# explicit precision: False
# num layers: 0
# [11/22/2023-22:17:49] [TRT] [W] onnx2trt_utils.cpp:374: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
# INFO:EngineBuilder:Network Description
# INFO:EngineBuilder:Input 'image_embeddings' with shape (1, 256, 64, 64) and dtype DataType.FLOAT
# INFO:EngineBuilder:Input 'point_coords' with shape (32, 1, 2) and dtype DataType.FLOAT
# INFO:EngineBuilder:Input 'point_labels' with shape (32, 1) and dtype DataType.FLOAT
# INFO:EngineBuilder:Input 'mask_input' with shape (1, 1, 256, 256) and dtype DataType.FLOAT
# INFO:EngineBuilder:Input 'has_mask_input' with shape (1,) and dtype DataType.FLOAT
# INFO:EngineBuilder:Output 'stacked_output' with shape (32, 4, 65537) and dtype DataType.FLOAT
# /NFS/home/nicole/gaze_sam/integration/build_fp16_decoder.py:103: DeprecationWarning: Use network created with NetworkDefinitionCreationFlag::EXPLICIT_BATCH flag instead.
#   self.builder.max_batch_size = self.batch_size
# INFO:EngineBuilder:Builder max batchsize: 1
# encoder path: engines/vit/encoder_k9_fp32_trt8.6.engine
# [11/22/2023-22:17:49] [TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
# network num layers in create engine: 771
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# layer contains layerNorm
# num inputs into the model: 5
# inputs: [<tensorrt.tensorrt.ITensor object at 0x7f2306eeb830>, <tensorrt.tensorrt.ITensor object at 0x7f2306065830>, <tensorrt.tensorrt.ITensor object at 0x7f2306066970>, <tensorrt.tensorrt.ITensor object at 0x7f2306065c70>, <tensorrt.tensorrt.ITensor object at 0x7f2306c891f0>]
# /NFS/home/nicole/gaze_sam/integration/build_fp16_decoder.py:179: DeprecationWarning: Use build_serialized_network instead.
#   with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
# [11/22/2023-22:17:49] [TRT] [W] Detected layernorm nodes in FP16: , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , 
# [11/22/2023-22:17:49] [TRT] [W] Running layernorm after self-attention in FP16 may cause overflow. Exporting the model to the latest available ONNX opset (later than opset 17) to use the INormalizationLayer, or forcing layernorm layers to run in FP32 precision can help with preserving accuracy.
# [11/22/2023-22:17:49] [TRT] [I] Graph optimization time: 0.203289 seconds.
# [11/22/2023-22:17:49] [TRT] [I] Local timing cache in use. Profiling results in this builder pass will not be stored.
# [11/22/2023-22:20:57] [TRT] [I] Detected 5 inputs and 3 output network tensors.
# [11/22/2023-22:20:58] [TRT] [I] Total Host Persistent Memory: 36848
# [11/22/2023-22:20:58] [TRT] [I] Total Device Persistent Memory: 356864
# [11/22/2023-22:20:58] [TRT] [I] Total Scratch Memory: 588382208
# [11/22/2023-22:20:58] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 71 MiB, GPU 1422 MiB
# [11/22/2023-22:20:58] [TRT] [I] [BlockAssignment] Started assigning block shifts. This will take 31 steps to complete.
# [11/22/2023-22:20:58] [TRT] [I] [BlockAssignment] Algorithm ShiftNTopDown took 0.704972ms to assign 8 blocks to 31 nodes requiring 924352512 bytes.
# [11/22/2023-22:20:58] [TRT] [I] Total Activation Memory: 924352512
# [11/22/2023-22:20:58] [TRT] [W] TensorRT encountered issues when converting weights between types and that could affect accuracy.
# [11/22/2023-22:20:58] [TRT] [W] If this is not the desired behavior, please modify the weights or retrain with regularization to adjust the magnitude of the weights.
# [11/22/2023-22:20:58] [TRT] [W] Check verbose logs for the list of affected weights.
# [11/22/2023-22:20:58] [TRT] [W] - 80 weights are affected by this issue: Detected subnormal FP16 values.
# [11/22/2023-22:20:58] [TRT] [W] - 12 weights are affected by this issue: Detected values less than smallest positive FP16 subnormal value and converted them to the FP16 minimum subnormalized value.
# [11/22/2023-22:20:58] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +206, now: CPU 0, GPU 464 (MiB)
# INFO:EngineBuilder:Serializing engine to file: /NFS/home/nicole/gaze_sam/integration/engines/vit/decoder_fp16_k9.engine
# (efficientvit) nicole@k9:~/gaze_sam/integration$ python combo.py

# ~~~ ITER 1 with file ../base_imgs/gum.png ~~~
# loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# encoder preprocess time: 0.020961761474609375
# prep encoder time: 0.04110240936279297
# prep decoder time: 0.013219118118286133
# iou access time: 4.76837158203125e-07
# low res mask access time: 0.0
# prep encoder time: 0.0010406970977783203
# prep decoder time: 0.0007693767547607422
# iou access time: 0.0
# low res mask access time: 2.384185791015625e-07
# output shape: (2,)
# Image Size: W=1280, H=720
# output shape: (2,)
# num crop boxes: 1
# 			crop preprocess time: 1.6689300537109375e-06
# image shape after preprocess: torch.Size([1, 3, 512, 512])
# features after passing through encoder: torch.Size([1, 256, 64, 64])
# 			MASK ENCODER TIME: 0.009433746337890625
# 			point preprocessing time: 2.384185791015625e-07
# 				batch preprocess time: 0.004535198211669922
# 				BATCH DECODER TIME: 0.003426074981689453
# 					convert to MaskData class: 7.009506225585938e-05
# 					iou filtering time: 0.013921737670898438
# 					stability score filtering time: 0.0021071434020996094
# 					thresholding time: 0.0013904571533203125
# 					box filtering time: 2.384185791015625e-07
# 					mask uncrop time: 2.6226043701171875e-06
# 					rle compression time: 3.337860107421875e-06
# 				batch filtering time: 0.01749563217163086
# 			batch process time: 0.02551126480102539
# num iou preds before nms: torch.Size([30])
# 			batch nms time: 0.0012328624725341797
# num iou preds after nms: torch.Size([12])
# 			uncrop time: 0.00013971328735351562
# 		crop process time: 0.03712320327758789
# 		duplicate crop removal time: 0.0038449764251708984
# mask data segmentations len: 12
# 	mask generation time: 0.041017770767211914
# 	postprocess time: 2.384185791015625e-07
# 	rle encoding time: 9.5367431640625e-06
# 	write MaskData: 0.00012063980102539062
# number of bounding boxes: 18


# ~ extracting one mask ~
# num anns: 12
# img.shape: (720, 1280, 3)
# get best max: 1700709765.361639
# find intersection point: 2.384185791015625e-07
# set mask: 0.0025315284729003906
# draw marker: 3.743171691894531e-05
# draw line mask + best bounding box: 2.4318695068359375e-05

# encoder/decoder priming run: 0.5496976375579834
# all gaze engines priming run: 0.11068892478942871
# yolo priming run: 1.0778508186340332

# load img: 0.06624293327331543
# resize img: 1.7402348518371582
# generate masks: 0.04120516777038574
# detect face (primed): 0.0020737648010253906
# smooth + extract face (primed): 4.291534423828125e-05
# detect landmark (primed): 0.0007708072662353516
# smooth landmark (primed): 0.0005521774291992188
# detect gaze (primed): 0.003420114517211914
# smooth gaze (primed): 1.1444091796875e-05
# visualize gaze: 0.0006690025329589844
# create plots: 5.0067901611328125e-06
# get gaze mask: 0.00033354759216308594
# prep yolo img: 0.001462697982788086
# yolo pred: 0.001249551773071289
# total yolo: 0.002712249755859375
# draw and get yolo boxes: 0.004037380218505859
# segment one mask: 0.004014492034912109

# display image: 0.016095876693725586
# save to file (out/quantized_yolo/1700709777.6106725.png): 0.7196424007415771
# non-load total: 0.05985236167907715
# load total: 1.8875885009765625


# ~~~ ITER 2 with file ../base_imgs/help.png ~~~
# loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# encoder preprocess time: 0.020724058151245117
# prep encoder time: 0.001767873764038086
# prep decoder time: 0.00090789794921875
# iou access time: 2.384185791015625e-07
# low res mask access time: 0.0
# prep encoder time: 0.0009644031524658203
# prep decoder time: 0.0004982948303222656
# iou access time: 4.76837158203125e-07
# low res mask access time: 0.0
# output shape: (2,)
# Image Size: W=1280, H=720
# output shape: (2,)
# num crop boxes: 1
# 			crop preprocess time: 1.6689300537109375e-06
# image shape after preprocess: torch.Size([1, 3, 512, 512])
# features after passing through encoder: torch.Size([1, 256, 64, 64])
# 			MASK ENCODER TIME: 0.009389400482177734
# 			point preprocessing time: 2.384185791015625e-07
# 				batch preprocess time: 0.004504680633544922
# 				BATCH DECODER TIME: 0.0013298988342285156
# 					convert to MaskData class: 4.696846008300781e-05
# 					iou filtering time: 0.014196157455444336
# 					stability score filtering time: 0.0029218196868896484
# 					thresholding time: 0.0014264583587646484
# 					box filtering time: 2.384185791015625e-07
# 					mask uncrop time: 2.384185791015625e-06
# 					rle compression time: 2.384185791015625e-06
# 				batch filtering time: 0.018596410751342773
# 			batch process time: 0.024477481842041016
# num iou preds before nms: torch.Size([60])
# 			batch nms time: 0.0006449222564697266
# num iou preds after nms: torch.Size([6])
# 			uncrop time: 0.00010180473327636719
# 		crop process time: 0.03525543212890625
# 		duplicate crop removal time: 0.0007293224334716797
# mask data segmentations len: 6
# 	mask generation time: 0.03602933883666992
# 	postprocess time: 7.152557373046875e-07
# 	rle encoding time: 5.245208740234375e-06
# 	write MaskData: 6.4849853515625e-05
# number of bounding boxes: 10


# ~ extracting one mask ~
# num anns: 6
# img.shape: (720, 1280, 3)
# no box intersection
# [   0. 6179.    0. 9622. 5890.  987.]
# get best max: 1700709784.7857409
# find intersection point: 2.384185791015625e-07
# set mask: 0.0025339126586914062
# draw marker: 4.0531158447265625e-05
# draw line mask + best bounding box: 7.62939453125e-06

# encoder/decoder priming run: 0.5240533351898193
# all gaze engines priming run: 0.09410381317138672
# yolo priming run: 1.0835785865783691

# load img: 0.04022407531738281
# resize img: 1.702484130859375
# generate masks: 0.03614377975463867
# detect face (primed): 0.0034575462341308594
# smooth + extract face (primed): 4.57763671875e-05
# detect landmark (primed): 0.0008983612060546875
# smooth landmark (primed): 0.0005536079406738281
# detect gaze (primed): 0.003542661666870117
# smooth gaze (primed): 1.0728836059570312e-05
# visualize gaze: 0.0006320476531982422
# create plots: 5.245208740234375e-06
# get gaze mask: 0.0001742839813232422
# prep yolo img: 0.0014543533325195312
# yolo pred: 0.001184701919555664
# total yolo: 0.0026390552520751953
# draw and get yolo boxes: 0.003937959671020508
# segment one mask: 0.003333568572998047

# display image: 0.0019867420196533203
# save to file (out/quantized_yolo/1700709782.120242.png): 0.8729643821716309
# non-load total: 0.05537891387939453
# load total: 0.8704750537872314


# ~~~ ITER 3 with file ../base_imgs/pen.png ~~~
# loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# encoder preprocess time: 0.020969867706298828
# prep encoder time: 0.0017938613891601562
# prep decoder time: 0.00091552734375
# iou access time: 2.384185791015625e-07
# low res mask access time: 0.0
# prep encoder time: 0.0009655952453613281
# prep decoder time: 0.0004982948303222656
# iou access time: 0.0
# low res mask access time: 0.0
# output shape: (2,)
# Image Size: W=1280, H=720
# output shape: (2,)
# num crop boxes: 1
# 			crop preprocess time: 1.1920928955078125e-06
# image shape after preprocess: torch.Size([1, 3, 512, 512])
# features after passing through encoder: torch.Size([1, 256, 64, 64])
# 			MASK ENCODER TIME: 0.009765148162841797
# 			point preprocessing time: 4.76837158203125e-07
# 				batch preprocess time: 0.0044269561767578125
# 				BATCH DECODER TIME: 0.0015616416931152344
# 					convert to MaskData class: 5.269050598144531e-05
# 					iou filtering time: 0.014139890670776367
# 					stability score filtering time: 0.0028641223907470703
# 					thresholding time: 0.0004742145538330078
# 					box filtering time: 2.384185791015625e-07
# 					mask uncrop time: 3.814697265625e-06
# 					rle compression time: 2.384185791015625e-06
# 				batch filtering time: 0.017537355422973633
# 			batch process time: 0.023578643798828125
# num iou preds before nms: torch.Size([66])
# 			batch nms time: 0.0006716251373291016
# num iou preds after nms: torch.Size([13])
# 			uncrop time: 0.00011038780212402344
# 		crop process time: 0.03482842445373535
# 		duplicate crop removal time: 0.0013222694396972656
# mask data segmentations len: 13
# 	mask generation time: 0.036206960678100586
# 	postprocess time: 4.76837158203125e-07
# 	rle encoding time: 5.4836273193359375e-06
# 	write MaskData: 0.00014352798461914062
# number of bounding boxes: 18


# ~ extracting one mask ~
# num anns: 13
# img.shape: (720, 1280, 3)
# get best max: 1700709787.2914755
# find intersection point: 0.0
# set mask: 0.006138801574707031
# draw marker: 5.435943603515625e-05
# draw line mask + best bounding box: 2.4080276489257812e-05

# encoder/decoder priming run: 0.4940316677093506
# all gaze engines priming run: 0.09408998489379883
# yolo priming run: 1.0736079216003418

# load img: 0.07539796829223633
# resize img: 1.6626064777374268
# generate masks: 0.036429405212402344
# detect face (primed): 0.0035300254821777344
# smooth + extract face (primed): 4.7206878662109375e-05
# detect landmark (primed): 0.00086212158203125
# smooth landmark (primed): 0.0005414485931396484
# detect gaze (primed): 0.003579854965209961
# smooth gaze (primed): 1.1682510375976562e-05
# visualize gaze: 0.0006542205810546875
# create plots: 5.4836273193359375e-06
# get gaze mask: 0.0003948211669921875
# prep yolo img: 0.0014200210571289062
# yolo pred: 0.0012104511260986328
# total yolo: 0.002630472183227539
# draw and get yolo boxes: 0.004038572311401367
# segment one mask: 0.007850408554077148

# display image: 0.002588987350463867
# save to file (out/quantized_yolo/1700709785.6951134.png): 1.1136555671691895
# non-load total: 0.06058001518249512
# load total: 0.8045165538787842


# ~~~ ITER 4 with file ../base_imgs/psycho.png ~~~
# loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# encoder preprocess time: 0.020186424255371094
# prep encoder time: 0.0016248226165771484
# prep decoder time: 0.0008533000946044922
# iou access time: 2.384185791015625e-07
# low res mask access time: 0.0
# prep encoder time: 0.0009648799896240234
# prep decoder time: 0.0004999637603759766
# iou access time: 0.0
# low res mask access time: 2.384185791015625e-07
# output shape: (2,)
# Image Size: W=1280, H=720
# output shape: (2,)
# num crop boxes: 1
# 			crop preprocess time: 1.6689300537109375e-06
# image shape after preprocess: torch.Size([1, 3, 512, 512])
# features after passing through encoder: torch.Size([1, 256, 64, 64])
# 			MASK ENCODER TIME: 0.00925755500793457
# 			point preprocessing time: 4.76837158203125e-07
# 				batch preprocess time: 0.0044934749603271484
# 				BATCH DECODER TIME: 0.001245737075805664
# 					convert to MaskData class: 4.5299530029296875e-05
# 					iou filtering time: 0.014275550842285156
# 					stability score filtering time: 0.0027375221252441406
# 					thresholding time: 0.00044798851013183594
# 					box filtering time: 2.384185791015625e-07
# 					mask uncrop time: 2.384185791015625e-06
# 					rle compression time: 2.384185791015625e-06
# 				batch filtering time: 0.017511367797851562
# 			batch process time: 0.02329707145690918
# num iou preds before nms: torch.Size([68])
# 			batch nms time: 0.0006237030029296875
# num iou preds after nms: torch.Size([12])
# 			uncrop time: 0.00010752677917480469
# 		crop process time: 0.03390622138977051
# 		duplicate crop removal time: 0.0012326240539550781
# mask data segmentations len: 12
# 	mask generation time: 0.03518486022949219
# 	postprocess time: 4.76837158203125e-07
# 	rle encoding time: 5.9604644775390625e-06
# 	write MaskData: 0.00012302398681640625
# number of bounding boxes: 13


# ~ extracting one mask ~
# num anns: 12
# img.shape: (720, 1280, 3)
# get best max: 1700709792.0520773
# find intersection point: 2.384185791015625e-07
# set mask: 0.006318569183349609
# draw marker: 3.8623809814453125e-05
# draw line mask + best bounding box: 2.6464462280273438e-05

# encoder/decoder priming run: 0.4994821548461914
# all gaze engines priming run: 0.0941627025604248
# yolo priming run: 1.0639774799346924

# load img: 0.07607674598693848
# resize img: 1.6829509735107422
# generate masks: 0.03536272048950195
# detect face (primed): 0.003192901611328125
# smooth + extract face (primed): 4.172325134277344e-05
# detect landmark (primed): 0.0008306503295898438
# smooth landmark (primed): 0.0006113052368164062
# detect gaze (primed): 0.003408670425415039
# smooth gaze (primed): 1.1682510375976562e-05
# visualize gaze: 0.0006079673767089844
# create plots: 5.245208740234375e-06
# get gaze mask: 0.0003304481506347656
# prep yolo img: 0.0013518333435058594
# yolo pred: 0.0011703968048095703
# total yolo: 0.0025222301483154297
# draw and get yolo boxes: 0.004045963287353516
# segment one mask: 0.007635593414306641

# display image: 0.002007722854614258
# save to file (out/quantized_yolo/1700709789.4480853.png): 1.3506760597229004
# non-load total: 0.05861258506774902
# load total: 0.7933652400970459


# ~~~ ITER 5 with file ../base_imgs/workpls_v2.png ~~~
# loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# encoder preprocess time: 0.020039796829223633
# prep encoder time: 0.0016126632690429688
# prep decoder time: 0.0008709430694580078
# iou access time: 2.384185791015625e-07
# low res mask access time: 0.0
# prep encoder time: 0.0009808540344238281
# prep decoder time: 0.0005052089691162109
# iou access time: 0.0
# low res mask access time: 2.384185791015625e-07
# output shape: (2,)
# Image Size: W=1280, H=720
# output shape: (2,)
# num crop boxes: 1
# 			crop preprocess time: 1.6689300537109375e-06
# image shape after preprocess: torch.Size([1, 3, 512, 512])
# features after passing through encoder: torch.Size([1, 256, 64, 64])
# 			MASK ENCODER TIME: 0.00989675521850586
# 			point preprocessing time: 4.76837158203125e-07
# 				batch preprocess time: 0.004496335983276367
# 				BATCH DECODER TIME: 0.0013127326965332031
# 					convert to MaskData class: 4.792213439941406e-05
# 					iou filtering time: 0.014458656311035156
# 					stability score filtering time: 0.003874063491821289
# 					thresholding time: 0.00044155120849609375
# 					box filtering time: 2.384185791015625e-07
# 					mask uncrop time: 2.384185791015625e-06
# 					rle compression time: 1.9073486328125e-06
# 				batch filtering time: 0.018826723098754883
# 			batch process time: 0.024744033813476562
# num iou preds before nms: torch.Size([101])
# 			batch nms time: 0.0008876323699951172
# num iou preds after nms: torch.Size([8])
# 			uncrop time: 0.00011396408081054688
# 		crop process time: 0.036269426345825195
# 		duplicate crop removal time: 0.0015370845794677734
# mask data segmentations len: 8
# 	mask generation time: 0.0378565788269043
# 	postprocess time: 4.76837158203125e-07
# 	rle encoding time: 7.3909759521484375e-06
# 	write MaskData: 9.059906005859375e-05
# number of bounding boxes: 2


# ~ extracting one mask ~
# num anns: 8
# img.shape: (720, 1280, 3)
# get best max: 1700709796.0400052
# find intersection point: 2.384185791015625e-07
# set mask: 0.0024635791778564453
# draw marker: 3.361701965332031e-05
# draw line mask + best bounding box: 2.3365020751953125e-05

# encoder/decoder priming run: 0.49974536895751953
# all gaze engines priming run: 0.10013127326965332
# yolo priming run: 1.0680062770843506

# load img: 0.08376383781433105
# resize img: 1.6688482761383057
# generate masks: 0.03800821304321289
# detect face (primed): 0.003244638442993164
# smooth + extract face (primed): 4.0531158447265625e-05
# detect landmark (primed): 0.0007891654968261719
# smooth landmark (primed): 0.0005552768707275391
# detect gaze (primed): 0.0033736228942871094
# smooth gaze (primed): 1.239776611328125e-05
# visualize gaze: 0.0006232261657714844
# create plots: 5.245208740234375e-06
# get gaze mask: 0.0002715587615966797
# prep yolo img: 0.00147247314453125
# yolo pred: 0.001203298568725586
# total yolo: 0.002675771713256836
# draw and get yolo boxes: 0.003863811492919922
# segment one mask: 0.0031304359436035156

# display image: 0.0023775100708007812
# save to file (out/quantized_yolo/1700709793.4432893.png): 1.7450013160705566
# non-load total: 0.05659914016723633
# load total: 0.7905194759368896


# ~~~ ITER 6 with file ../base_imgs/zz.png ~~~
# loading model: efficient_vit/assets/checkpoints/sam/l0.pt
# encoder preprocess time: 0.02026653289794922
# prep encoder time: 0.001547098159790039
# prep decoder time: 0.0008525848388671875
# iou access time: 4.76837158203125e-07
# low res mask access time: 2.384185791015625e-07
# prep encoder time: 0.0009715557098388672
# prep decoder time: 0.0005154609680175781
# iou access time: 2.384185791015625e-07
# low res mask access time: 0.0
# output shape: (2,)
# Image Size: W=1280, H=720
# output shape: (2,)
# num crop boxes: 1
# 			crop preprocess time: 1.430511474609375e-06
# image shape after preprocess: torch.Size([1, 3, 512, 512])
# features after passing through encoder: torch.Size([1, 256, 64, 64])
# 			MASK ENCODER TIME: 0.009355783462524414
# 			point preprocessing time: 4.76837158203125e-07
# 				batch preprocess time: 0.004503965377807617
# 				BATCH DECODER TIME: 0.0012290477752685547
# 					convert to MaskData class: 4.5299530029296875e-05
# 					iou filtering time: 0.014318704605102539
# 					stability score filtering time: 0.0029098987579345703
# 					thresholding time: 0.0004143714904785156
# 					box filtering time: 0.0
# 					mask uncrop time: 2.384185791015625e-06
# 					rle compression time: 2.384185791015625e-06
# 				batch filtering time: 0.017693042755126953
# 			batch process time: 0.023473262786865234
# num iou preds before nms: torch.Size([47])
# 			batch nms time: 0.0005538463592529297
# num iou preds after nms: torch.Size([6])
# 			uncrop time: 0.0001087188720703125
# 		crop process time: 0.034119367599487305
# 		duplicate crop removal time: 0.0006692409515380859
# mask data segmentations len: 6
# 	mask generation time: 0.03483295440673828
# 	postprocess time: 7.152557373046875e-07
# 	rle encoding time: 7.867813110351562e-06
# 	write MaskData: 6.818771362304688e-05
# number of bounding boxes: 11


# ~ extracting one mask ~
# num anns: 6
# img.shape: (720, 1280, 3)
# get best max: 1700709792.4775689
# find intersection point: 2.384185791015625e-07
# set mask: 0.0040283203125
# draw marker: 3.2901763916015625e-05
# draw line mask + best bounding box: 4.172325134277344e-05

# encoder/decoder priming run: 0.5352756977081299
# all gaze engines priming run: 0.09366917610168457
# yolo priming run: 1.0657100677490234

# load img: 0.07424187660217285
# resize img: 1.695589542388916
# generate masks: 0.03495383262634277
# detect face (primed): 0.003067493438720703
# smooth + extract face (primed): 4.267692565917969e-05
# detect landmark (primed): 0.0007982254028320312
# smooth landmark (primed): 0.0005595684051513672
# detect gaze (primed): 0.003391742706298828
# smooth gaze (primed): 1.1444091796875e-05
# visualize gaze: 0.0006165504455566406
# create plots: 5.245208740234375e-06
# get gaze mask: 0.0002548694610595703
# prep yolo img: 0.0013890266418457031
# yolo pred: 0.0011668205261230469
# total yolo: 0.00255584716796875
# draw and get yolo boxes: 0.0039806365966796875
# segment one mask: 0.004609823226928711

# display image: 0.001965045928955078
# save to file (out/quantized_yolo/1700709797.8229375.png): 1.8958489894866943
# non-load total: 0.054853200912475586
# load total: 0.834507942199707

# (efficientvit) nicole@k9:~/gaze_sam/integration$ trtexec --loadEngine=engines/vit/decoder_fp16_k9.engine
# -bash: /home/nicole/.local/TensorRT-8.6.1.6/bin: Is a directory
# (efficientvit) nicole@k9:~/gaze_sam/integration$ /home/nicole/.local/TensorRT-8.6.1.6/bin/trtexec --loadEngine=engines/vit/decoder_fp16_k9.engine
# &&&& RUNNING TensorRT.trtexec [TensorRT v8601] # /home/nicole/.local/TensorRT-8.6.1.6/bin/trtexec --loadEngine=engines/vit/decoder_fp16_k9.engine
# [11/22/2023-22:24:02] [I] === Model Options ===
# [11/22/2023-22:24:02] [I] Format: *
# [11/22/2023-22:24:02] [I] Model: 
# [11/22/2023-22:24:02] [I] Output:
# [11/22/2023-22:24:02] [I] === Build Options ===
# [11/22/2023-22:24:02] [I] Max batch: 1
# [11/22/2023-22:24:02] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default
# [11/22/2023-22:24:02] [I] minTiming: 1
# [11/22/2023-22:24:02] [I] avgTiming: 8
# [11/22/2023-22:24:02] [I] Precision: FP32
# [11/22/2023-22:24:02] [I] LayerPrecisions: 
# [11/22/2023-22:24:02] [I] Layer Device Types: 
# [11/22/2023-22:24:02] [I] Calibration: 
# [11/22/2023-22:24:02] [I] Refit: Disabled
# [11/22/2023-22:24:02] [I] Version Compatible: Disabled
# [11/22/2023-22:24:02] [I] TensorRT runtime: full
# [11/22/2023-22:24:02] [I] Lean DLL Path: 
# [11/22/2023-22:24:02] [I] Tempfile Controls: { in_memory: allow, temporary: allow }
# [11/22/2023-22:24:02] [I] Exclude Lean Runtime: Disabled
# [11/22/2023-22:24:02] [I] Sparsity: Disabled
# [11/22/2023-22:24:02] [I] Safe mode: Disabled
# [11/22/2023-22:24:02] [I] Build DLA standalone loadable: Disabled
# [11/22/2023-22:24:02] [I] Allow GPU fallback for DLA: Disabled
# [11/22/2023-22:24:02] [I] DirectIO mode: Disabled
# [11/22/2023-22:24:02] [I] Restricted mode: Disabled
# [11/22/2023-22:24:02] [I] Skip inference: Disabled
# [11/22/2023-22:24:02] [I] Save engine: 
# [11/22/2023-22:24:02] [I] Load engine: engines/vit/decoder_fp16_k9.engine
# [11/22/2023-22:24:02] [I] Profiling verbosity: 0
# [11/22/2023-22:24:02] [I] Tactic sources: Using default tactic sources
# [11/22/2023-22:24:02] [I] timingCacheMode: local
# [11/22/2023-22:24:02] [I] timingCacheFile: 
# [11/22/2023-22:24:02] [I] Heuristic: Disabled
# [11/22/2023-22:24:02] [I] Preview Features: Use default preview flags.
# [11/22/2023-22:24:02] [I] MaxAuxStreams: -1
# [11/22/2023-22:24:02] [I] BuilderOptimizationLevel: -1
# [11/22/2023-22:24:02] [I] Input(s)s format: fp32:CHW
# [11/22/2023-22:24:02] [I] Output(s)s format: fp32:CHW
# [11/22/2023-22:24:02] [I] Input build shapes: model
# [11/22/2023-22:24:02] [I] Input calibration shapes: model
# [11/22/2023-22:24:02] [I] === System Options ===
# [11/22/2023-22:24:02] [I] Device: 0
# [11/22/2023-22:24:02] [I] DLACore: 
# [11/22/2023-22:24:02] [I] Plugins:
# [11/22/2023-22:24:02] [I] setPluginsToSerialize:
# [11/22/2023-22:24:02] [I] dynamicPlugins:
# [11/22/2023-22:24:02] [I] ignoreParsedPluginLibs: 0
# [11/22/2023-22:24:02] [I] 
# [11/22/2023-22:24:02] [I] === Inference Options ===
# [11/22/2023-22:24:02] [I] Batch: 1
# [11/22/2023-22:24:02] [I] Input inference shapes: model
# [11/22/2023-22:24:02] [I] Iterations: 10
# [11/22/2023-22:24:02] [I] Duration: 3s (+ 200ms warm up)
# [11/22/2023-22:24:02] [I] Sleep time: 0ms
# [11/22/2023-22:24:02] [I] Idle time: 0ms
# [11/22/2023-22:24:02] [I] Inference Streams: 1
# [11/22/2023-22:24:02] [I] ExposeDMA: Disabled
# [11/22/2023-22:24:02] [I] Data transfers: Enabled
# [11/22/2023-22:24:02] [I] Spin-wait: Disabled
# [11/22/2023-22:24:02] [I] Multithreading: Disabled
# [11/22/2023-22:24:02] [I] CUDA Graph: Disabled
# [11/22/2023-22:24:02] [I] Separate profiling: Disabled
# [11/22/2023-22:24:02] [I] Time Deserialize: Disabled
# [11/22/2023-22:24:02] [I] Time Refit: Disabled
# [11/22/2023-22:24:02] [I] NVTX verbosity: 0
# [11/22/2023-22:24:02] [I] Persistent Cache Ratio: 0
# [11/22/2023-22:24:02] [I] Inputs:
# [11/22/2023-22:24:02] [I] === Reporting Options ===
# [11/22/2023-22:24:02] [I] Verbose: Disabled
# [11/22/2023-22:24:02] [I] Averages: 10 inferences
# [11/22/2023-22:24:02] [I] Percentiles: 90,95,99
# [11/22/2023-22:24:02] [I] Dump refittable layers:Disabled
# [11/22/2023-22:24:02] [I] Dump output: Disabled
# [11/22/2023-22:24:02] [I] Profile: Disabled
# [11/22/2023-22:24:02] [I] Export timing to JSON file: 
# [11/22/2023-22:24:02] [I] Export output to JSON file: 
# [11/22/2023-22:24:02] [I] Export profile to JSON file: 
# [11/22/2023-22:24:02] [I] 
# [11/22/2023-22:24:02] [I] === Device Information ===
# [11/22/2023-22:24:02] [I] Selected Device: NVIDIA GeForce RTX 3090
# [11/22/2023-22:24:02] [I] Compute Capability: 8.6
# [11/22/2023-22:24:02] [I] SMs: 82
# [11/22/2023-22:24:02] [I] Device Global Memory: 24259 MiB
# [11/22/2023-22:24:02] [I] Shared Memory per SM: 100 KiB
# [11/22/2023-22:24:02] [I] Memory Bus Width: 384 bits (ECC disabled)
# [11/22/2023-22:24:02] [I] Application Compute Clock Rate: 1.695 GHz
# [11/22/2023-22:24:02] [I] Application Memory Clock Rate: 9.751 GHz
# [11/22/2023-22:24:02] [I] 
# [11/22/2023-22:24:02] [I] Note: The application clock rates do not reflect the actual clock rates that the GPU is currently running at.
# [11/22/2023-22:24:02] [I] 
# [11/22/2023-22:24:02] [I] TensorRT version: 8.6.1
# [11/22/2023-22:24:02] [I] Loading standard plugins
# [11/22/2023-22:24:02] [I] Engine loaded in 0.156547 sec.
# [11/22/2023-22:24:03] [I] [TRT] Loaded engine size: 208 MiB
# [11/22/2023-22:24:03] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +206, now: CPU 0, GPU 206 (MiB)
# [11/22/2023-22:24:03] [I] Engine deserialized in 0.360613 sec.
# [11/22/2023-22:24:03] [I] [TRT] [MS] Running engine with multi stream info
# [11/22/2023-22:24:03] [I] [TRT] [MS] Number of aux streams is 4
# [11/22/2023-22:24:03] [I] [TRT] [MS] Number of total worker streams is 5
# [11/22/2023-22:24:03] [I] [TRT] [MS] The main stream provided by execute/enqueue calls is the first worker stream
# [11/22/2023-22:24:03] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +882, now: CPU 0, GPU 1088 (MiB)
# [11/22/2023-22:24:03] [W] [TRT] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
# [11/22/2023-22:24:03] [I] Setting persistentCacheLimit to 0 bytes.
# [11/22/2023-22:24:03] [I] Using random values for input image_embeddings
# [11/22/2023-22:24:03] [I] Input binding for image_embeddings with dimensions 1x256x64x64 is created.
# [11/22/2023-22:24:03] [I] Using random values for input point_coords
# [11/22/2023-22:24:03] [I] Input binding for point_coords with dimensions 32x1x2 is created.
# [11/22/2023-22:24:03] [I] Using random values for input point_labels
# [11/22/2023-22:24:03] [I] Input binding for point_labels with dimensions 32x1 is created.
# [11/22/2023-22:24:03] [I] Using random values for input mask_input
# [11/22/2023-22:24:03] [I] Input binding for mask_input with dimensions 1x1x256x256 is created.
# [11/22/2023-22:24:03] [I] Using random values for input has_mask_input
# [11/22/2023-22:24:03] [I] Input binding for has_mask_input with dimensions 1 is created.
# [11/22/2023-22:24:03] [I] Output binding for stacked_output with dimensions 32x4x65537 is created.
# [11/22/2023-22:24:03] [I] Starting inference
# [11/22/2023-22:24:06] [I] Warmup completed 17 queries over 200 ms
# [11/22/2023-22:24:06] [I] Timing trace has 251 queries over 3.03841 s
# [11/22/2023-22:24:06] [I] 
# [11/22/2023-22:24:06] [I] === Trace details ===
# [11/22/2023-22:24:06] [I] Trace averages of 10 runs:
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 11.9917 ms - Host latency: 15.1891 ms (enqueue 0.665701 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0003 ms - Host latency: 15.1583 ms (enqueue 0.690894 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 11.9879 ms - Host latency: 15.1578 ms (enqueue 0.770508 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 11.9796 ms - Host latency: 15.1522 ms (enqueue 0.723834 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 11.9842 ms - Host latency: 15.1602 ms (enqueue 0.726459 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.105 ms - Host latency: 15.3127 ms (enqueue 0.773816 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.1897 ms - Host latency: 15.3672 ms (enqueue 0.721118 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.114 ms - Host latency: 15.3146 ms (enqueue 0.728027 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0336 ms - Host latency: 15.2089 ms (enqueue 0.719775 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0207 ms - Host latency: 15.1979 ms (enqueue 0.724097 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.041 ms - Host latency: 15.248 ms (enqueue 0.720496 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 11.9982 ms - Host latency: 15.181 ms (enqueue 0.721069 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0067 ms - Host latency: 15.1926 ms (enqueue 0.720972 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0836 ms - Host latency: 15.2712 ms (enqueue 0.723669 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.1644 ms - Host latency: 15.3891 ms (enqueue 0.726282 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0967 ms - Host latency: 15.2758 ms (enqueue 0.725012 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0568 ms - Host latency: 15.2394 ms (enqueue 0.721313 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.04 ms - Host latency: 15.2239 ms (enqueue 0.721094 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0059 ms - Host latency: 15.195 ms (enqueue 0.721069 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0419 ms - Host latency: 15.2719 ms (enqueue 0.719971 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0552 ms - Host latency: 15.2406 ms (enqueue 0.721558 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.0632 ms - Host latency: 15.2547 ms (enqueue 0.726587 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.116 ms - Host latency: 15.3061 ms (enqueue 0.726025 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.1294 ms - Host latency: 15.3155 ms (enqueue 0.721899 ms)
# [11/22/2023-22:24:06] [I] Average on 10 runs - GPU latency: 12.1027 ms - Host latency: 15.333 ms (enqueue 0.719629 ms)
# [11/22/2023-22:24:06] [I] 
# [11/22/2023-22:24:06] [I] === Performance summary ===
# [11/22/2023-22:24:06] [I] Throughput: 82.609 qps
# [11/22/2023-22:24:06] [I] Latency: min = 14.9072 ms, max = 15.7528 ms, mean = 15.2449 ms, median = 15.2222 ms, percentile(90%) = 15.3647 ms, percentile(95%) = 15.387 ms, percentile(99%) = 15.6807 ms
# [11/22/2023-22:24:06] [I] Enqueue Time: min = 0.66214 ms, max = 1.21698 ms, mean = 0.723226 ms, median = 0.721191 ms, percentile(90%) = 0.731201 ms, percentile(95%) = 0.736816 ms, percentile(99%) = 0.776337 ms
# [11/22/2023-22:24:06] [I] H2D Latency: min = 0.245117 ms, max = 0.293945 ms, mean = 0.26085 ms, median = 0.260864 ms, percentile(90%) = 0.266602 ms, percentile(95%) = 0.268311 ms, percentile(99%) = 0.27301 ms
# [11/22/2023-22:24:06] [I] GPU Compute Time: min = 11.9439 ms, max = 12.245 ms, mean = 12.0562 ms, median = 12.0352 ms, percentile(90%) = 12.1641 ms, percentile(95%) = 12.1929 ms, percentile(99%) = 12.2143 ms
# [11/22/2023-22:24:06] [I] D2H Latency: min = 2.6333 ms, max = 3.34326 ms, mean = 2.92788 ms, median = 2.91956 ms, percentile(90%) = 2.93384 ms, percentile(95%) = 2.93945 ms, percentile(99%) = 3.33289 ms
# [11/22/2023-22:24:06] [I] Total Host Walltime: 3.03841 s
# [11/22/2023-22:24:06] [I] Total GPU Compute Time: 3.0261 s
# [11/22/2023-22:24:06] [I] Explanations of the performance metrics are printed in the verbose logs.
# [11/22/2023-22:24:06] [I] 
# &&&& PASSED TensorRT.trtexec [TensorRT v8601] # /home/nicole/.local/TensorRT-8.6.1.6/bin/trtexec --loadEngine=engines/vit/decoder_fp16_k9.engine
# (efficientvit) nicole@k9:~/gaze_sam/integration$ 
# """

res = """
python combo.py

~~~ ITER 1 with file ../base_imgs/gum.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.020859956741333008
prep encoder time: 0.04155588150024414
prep decoder time: 0.0011665821075439453
iou access time: 2.384185791015625e-07
low res mask access time: 0.0
prep encoder time: 0.0010533332824707031
prep decoder time: 0.003955841064453125
iou access time: 0.0
low res mask access time: 0.0
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.6689300537109375e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.010679483413696289
			point preprocessing time: 4.76837158203125e-07
				batch preprocess time: 0.004416704177856445
				BATCH DECODER TIME: 0.004748821258544922
					convert to MaskData class: 9.775161743164062e-05
					iou filtering time: 0.044394493103027344
					stability score filtering time: 0.0023245811462402344
					thresholding time: 0.004581928253173828
					box filtering time: 4.76837158203125e-07
					mask uncrop time: 5.4836273193359375e-06
					rle compression time: 5.0067901611328125e-06
				batch filtering time: 0.05140972137451172
			batch process time: 0.06079864501953125
num iou preds before nms: torch.Size([29])
			batch nms time: 0.001895904541015625
num iou preds after nms: torch.Size([12])
			uncrop time: 0.00017333030700683594
		crop process time: 0.0745859146118164
		duplicate crop removal time: 0.004361629486083984
mask data segmentations len: 12
	mask generation time: 0.0790102481842041
	postprocess time: 4.76837158203125e-07
	rle encoding time: 7.3909759521484375e-06
	write MaskData: 0.0001556873321533203
number of bounding boxes: 18


~ extracting one mask ~
num anns: 12
img.shape: (720, 1280, 3)
get best max: 1700710156.4583623
find intersection point: 2.384185791015625e-07
set mask: 0.002589702606201172
draw marker: 4.029273986816406e-05
draw line mask + best bounding box: 3.075599670410156e-05

encoder/decoder priming run: 0.5055804252624512
all gaze engines priming run: 0.11021018028259277
yolo priming run: 1.0838913917541504

load img: 0.06577634811401367
resize img: 1.7005407810211182
generate masks: 0.07925009727478027
detect face (primed): 0.003175497055053711
smooth + extract face (primed): 5.221366882324219e-05
detect landmark (primed): 0.000990152359008789
smooth landmark (primed): 0.0005848407745361328
detect gaze (primed): 0.0041065216064453125
smooth gaze (primed): 1.33514404296875e-05
visualize gaze: 0.000850677490234375
create plots: 7.3909759521484375e-06
get gaze mask: 0.0003790855407714844
prep yolo img: 0.0017685890197753906
yolo pred: 0.0018143653869628906
total yolo: 0.0035829544067382812
draw and get yolo boxes: 0.0038831233978271484
segment one mask: 0.0041961669921875

display image: 0.02155160903930664
save to file (out/quantized_yolo/1700710168.7459743.png): 0.729905366897583
non-load total: 0.10107994079589844
load total: 1.8481941223144531


~~~ ITER 2 with file ../base_imgs/help.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.02300739288330078
prep encoder time: 0.002446889877319336
prep decoder time: 0.000812530517578125
iou access time: 2.384185791015625e-07
low res mask access time: 0.0
prep encoder time: 0.0009527206420898438
prep decoder time: 0.00047469139099121094
iou access time: 0.0
low res mask access time: 2.384185791015625e-07
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.6689300537109375e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.009571075439453125
			point preprocessing time: 2.384185791015625e-07
				batch preprocess time: 0.004415035247802734
				BATCH DECODER TIME: 0.0014514923095703125
					convert to MaskData class: 6.079673767089844e-05
					iou filtering time: 0.017739295959472656
					stability score filtering time: 0.002917766571044922
					thresholding time: 0.0016624927520751953
					box filtering time: 2.384185791015625e-07
					mask uncrop time: 1.1920928955078125e-05
					rle compression time: 2.6226043701171875e-06
				batch filtering time: 0.02239513397216797
			batch process time: 0.028321266174316406
num iou preds before nms: torch.Size([60])
			batch nms time: 0.0009472370147705078
num iou preds after nms: torch.Size([6])
			uncrop time: 0.0001442432403564453
		crop process time: 0.03992891311645508
		duplicate crop removal time: 0.0009412765502929688
mask data segmentations len: 6
	mask generation time: 0.0409550666809082
	postprocess time: 7.152557373046875e-07
	rle encoding time: 5.9604644775390625e-06
	write MaskData: 0.0001285076141357422
number of bounding boxes: 10


~ extracting one mask ~
num anns: 6
img.shape: (720, 1280, 3)
no box intersection
[   0. 6179.    0. 9621. 5891.  987.]
get best max: 1700710175.879741
find intersection point: 2.384185791015625e-07
set mask: 0.002625703811645508
draw marker: 6.0558319091796875e-05
draw line mask + best bounding box: 8.106231689453125e-06

encoder/decoder priming run: 0.4915802478790283
all gaze engines priming run: 0.09693551063537598
yolo priming run: 1.0797390937805176

load img: 0.04049372673034668
resize img: 1.6697120666503906
generate masks: 0.04123115539550781
detect face (primed): 0.003183603286743164
smooth + extract face (primed): 6.461143493652344e-05
detect landmark (primed): 0.0009319782257080078
smooth landmark (primed): 0.0006279945373535156
detect gaze (primed): 0.0036149024963378906
smooth gaze (primed): 1.2636184692382812e-05
visualize gaze: 0.0008678436279296875
create plots: 6.198883056640625e-06
get gaze mask: 0.0002713203430175781
prep yolo img: 0.0030281543731689453
yolo pred: 0.0015027523040771484
total yolo: 0.004530906677246094
draw and get yolo boxes: 0.0037300586700439453
segment one mask: 0.003977060317993164

display image: 0.005010843276977539
save to file (out/quantized_yolo/1700710173.2405443.png): 0.8887526988983154
non-load total: 0.06305575370788574
load total: 0.8693370819091797


~~~ ITER 3 with file ../base_imgs/pen.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.022286653518676758
prep encoder time: 0.001972198486328125
prep decoder time: 0.0008342266082763672
iou access time: 2.384185791015625e-07
low res mask access time: 0.0
prep encoder time: 0.000990152359008789
prep decoder time: 0.0006480216979980469
iou access time: 2.384185791015625e-07
low res mask access time: 9.5367431640625e-07
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.6689300537109375e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.01049041748046875
			point preprocessing time: 4.76837158203125e-07
				batch preprocess time: 0.0043370723724365234
				BATCH DECODER TIME: 0.0016217231750488281
					convert to MaskData class: 5.650520324707031e-05
					iou filtering time: 0.017866849899291992
					stability score filtering time: 0.002860546112060547
					thresholding time: 0.0005559921264648438
					box filtering time: 2.384185791015625e-07
					mask uncrop time: 4.291534423828125e-06
					rle compression time: 2.1457672119140625e-06
				batch filtering time: 0.021346569061279297
			batch process time: 0.02735614776611328
num iou preds before nms: torch.Size([65])
			batch nms time: 0.0007567405700683594
num iou preds after nms: torch.Size([12])
			uncrop time: 0.00011920928955078125
		crop process time: 0.03949093818664551
		duplicate crop removal time: 0.0017991065979003906
mask data segmentations len: 12
	mask generation time: 0.04136061668395996
	postprocess time: 7.152557373046875e-07
	rle encoding time: 6.198883056640625e-06
	write MaskData: 0.00014972686767578125
number of bounding boxes: 18


~ extracting one mask ~
num anns: 12
img.shape: (720, 1280, 3)
get best max: 1700710178.465844
find intersection point: 0.0
set mask: 0.006188631057739258
draw marker: 5.793571472167969e-05
draw line mask + best bounding box: 2.5272369384765625e-05

encoder/decoder priming run: 0.4995453357696533
all gaze engines priming run: 0.0964968204498291
yolo priming run: 1.0875964164733887

load img: 0.08264970779418945
resize img: 1.6849439144134521
generate masks: 0.041596174240112305
detect face (primed): 0.0038025379180908203
smooth + extract face (primed): 5.507469177246094e-05
detect landmark (primed): 0.0009338855743408203
smooth landmark (primed): 0.0005695819854736328
detect gaze (primed): 0.0038595199584960938
smooth gaze (primed): 1.2636184692382812e-05
visualize gaze: 0.0006897449493408203
create plots: 8.106231689453125e-06
get gaze mask: 0.00038933753967285156
prep yolo img: 0.00151824951171875
yolo pred: 0.0013134479522705078
total yolo: 0.002831697463989258
draw and get yolo boxes: 0.004004478454589844
segment one mask: 0.007820606231689453

display image: 0.0032651424407958984
save to file (out/quantized_yolo/1700710176.8167415.png): 1.1254689693450928
non-load total: 0.06657862663269043
load total: 0.822169303894043


~~~ ITER 4 with file ../base_imgs/psycho.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.021410703659057617
prep encoder time: 0.0018422603607177734
prep decoder time: 0.0007994174957275391
iou access time: 2.384185791015625e-07
low res mask access time: 0.0
prep encoder time: 0.0009522438049316406
prep decoder time: 0.0005006790161132812
iou access time: 2.384185791015625e-07
low res mask access time: 0.0
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.1920928955078125e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.010384082794189453
			point preprocessing time: 4.76837158203125e-07
				batch preprocess time: 0.00436854362487793
				BATCH DECODER TIME: 0.0012488365173339844
					convert to MaskData class: 4.887580871582031e-05
					iou filtering time: 0.01770806312561035
					stability score filtering time: 0.0027313232421875
					thresholding time: 0.00045680999755859375
					box filtering time: 0.0
					mask uncrop time: 2.384185791015625e-06
					rle compression time: 1.9073486328125e-06
				batch filtering time: 0.020949363708496094
			batch process time: 0.026617765426635742
num iou preds before nms: torch.Size([69])
			batch nms time: 0.0006291866302490234
num iou preds after nms: torch.Size([13])
			uncrop time: 0.00010848045349121094
		crop process time: 0.03839755058288574
		duplicate crop removal time: 0.0020525455474853516
mask data segmentations len: 13
	mask generation time: 0.04051375389099121
	postprocess time: 2.384185791015625e-07
	rle encoding time: 1.1682510375976562e-05
	write MaskData: 0.0001556873321533203
number of bounding boxes: 13


~ extracting one mask ~
num anns: 13
img.shape: (720, 1280, 3)
get best max: 1700710183.3231964
find intersection point: 2.384185791015625e-07
set mask: 0.006324291229248047
draw marker: 4.673004150390625e-05
draw line mask + best bounding box: 2.6941299438476562e-05

encoder/decoder priming run: 0.4910240173339844
all gaze engines priming run: 0.09463071823120117
yolo priming run: 1.0745248794555664

load img: 0.07718491554260254
resize img: 1.6615283489227295
generate masks: 0.04072833061218262
detect face (primed): 0.0033867359161376953
smooth + extract face (primed): 4.267692565917969e-05
detect landmark (primed): 0.0008521080017089844
smooth landmark (primed): 0.0005414485931396484
detect gaze (primed): 0.003572225570678711
smooth gaze (primed): 1.1444091796875e-05
visualize gaze: 0.0006036758422851562
create plots: 5.0067901611328125e-06
get gaze mask: 0.0003266334533691406
prep yolo img: 0.0014352798461914062
yolo pred: 0.0012617111206054688
total yolo: 0.002696990966796875
draw and get yolo boxes: 0.0039501190185546875
segment one mask: 0.007780551910400391

display image: 0.002521038055419922
save to file (out/quantized_yolo/1700710180.6369455.png): 1.356611967086792
non-load total: 0.06450247764587402
load total: 0.8909823894500732


~~~ ITER 5 with file ../base_imgs/workpls_v2.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.023494958877563477
prep encoder time: 0.0017578601837158203
prep decoder time: 0.0007770061492919922
iou access time: 2.384185791015625e-07
low res mask access time: 0.0
prep encoder time: 0.0009403228759765625
prep decoder time: 0.0004892349243164062
iou access time: 0.0
low res mask access time: 0.0
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.1920928955078125e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.010139942169189453
			point preprocessing time: 4.76837158203125e-07
				batch preprocess time: 0.0042552947998046875
				BATCH DECODER TIME: 0.001505136489868164
					convert to MaskData class: 5.6743621826171875e-05
					iou filtering time: 0.01778125762939453
					stability score filtering time: 0.0038573741912841797
					thresholding time: 0.0005071163177490234
					box filtering time: 2.384185791015625e-07
					mask uncrop time: 4.5299530029296875e-06
					rle compression time: 2.1457672119140625e-06
				batch filtering time: 0.02220940589904785
			batch process time: 0.028073787689208984
num iou preds before nms: torch.Size([101])
			batch nms time: 0.0007765293121337891
num iou preds after nms: torch.Size([8])
			uncrop time: 0.00011157989501953125
		crop process time: 0.039864540100097656
		duplicate crop removal time: 0.001224517822265625
mask data segmentations len: 8
	mask generation time: 0.041155338287353516
	postprocess time: 4.76837158203125e-07
	rle encoding time: 5.9604644775390625e-06
	write MaskData: 9.870529174804688e-05
number of bounding boxes: 2


~ extracting one mask ~
num anns: 8
img.shape: (720, 1280, 3)
get best max: 1700710187.327612
find intersection point: 2.384185791015625e-07
set mask: 0.0032706260681152344
draw marker: 4.363059997558594e-05
draw line mask + best bounding box: 2.7418136596679688e-05

encoder/decoder priming run: 0.4914262294769287
all gaze engines priming run: 0.09534859657287598
yolo priming run: 1.0931117534637451

load img: 0.07917046546936035
resize img: 1.6810946464538574
generate masks: 0.04133963584899902
detect face (primed): 0.003710031509399414
smooth + extract face (primed): 5.698204040527344e-05
detect landmark (primed): 0.0009288787841796875
smooth landmark (primed): 0.0005626678466796875
detect gaze (primed): 0.0038323402404785156
smooth gaze (primed): 1.2159347534179688e-05
visualize gaze: 0.000701904296875
create plots: 7.867813110351562e-06
get gaze mask: 0.0002818107604980469
prep yolo img: 0.0015482902526855469
yolo pred: 0.0013189315795898438
total yolo: 0.0028672218322753906
draw and get yolo boxes: 0.00371551513671875
segment one mask: 0.003979206085205078

display image: 0.0029985904693603516
save to file (out/quantized_yolo/1700710184.720476.png): 1.7432034015655518
non-load total: 0.06200218200683594
load total: 0.7887630462646484


~~~ ITER 6 with file ../base_imgs/zz.png ~~~
loading model: efficient_vit/assets/checkpoints/sam/l0.pt
encoder preprocess time: 0.02075791358947754
prep encoder time: 0.0017383098602294922
prep decoder time: 0.0008420944213867188
iou access time: 4.76837158203125e-07
low res mask access time: 0.0
prep encoder time: 0.0009543895721435547
prep decoder time: 0.00048804283142089844
iou access time: 2.384185791015625e-07
low res mask access time: 0.0
output shape: (2,)
Image Size: W=1280, H=720
output shape: (2,)
num crop boxes: 1
			crop preprocess time: 1.6689300537109375e-06
image shape after preprocess: torch.Size([1, 3, 512, 512])
features after passing through encoder: torch.Size([1, 256, 64, 64])
			MASK ENCODER TIME: 0.009992837905883789
			point preprocessing time: 9.5367431640625e-07
				batch preprocess time: 0.004239797592163086
				BATCH DECODER TIME: 0.0012483596801757812
					convert to MaskData class: 4.863739013671875e-05
					iou filtering time: 0.017644643783569336
					stability score filtering time: 0.0029001235961914062
					thresholding time: 0.0004937648773193359
					box filtering time: 2.384185791015625e-07
					mask uncrop time: 2.384185791015625e-06
					rle compression time: 2.384185791015625e-06
				batch filtering time: 0.02109217643737793
			batch process time: 0.026634693145751953
num iou preds before nms: torch.Size([47])
			batch nms time: 0.0006163120269775391
num iou preds after nms: torch.Size([6])
			uncrop time: 0.00010657310485839844
		crop process time: 0.03803682327270508
		duplicate crop removal time: 0.000926971435546875
mask data segmentations len: 6
	mask generation time: 0.039014339447021484
	postprocess time: 7.152557373046875e-07
	rle encoding time: 6.4373016357421875e-06
	write MaskData: 8.463859558105469e-05
number of bounding boxes: 11


~ extracting one mask ~
num anns: 6
img.shape: (720, 1280, 3)
get best max: 1700710183.7020614
find intersection point: 0.0
set mask: 0.004024505615234375
draw marker: 3.504753112792969e-05
draw line mask + best bounding box: 4.100799560546875e-05

encoder/decoder priming run: 0.4859755039215088
all gaze engines priming run: 0.09597897529602051
yolo priming run: 1.0867106914520264

load img: 0.07461023330688477
resize img: 1.6702003479003906
generate masks: 0.039153099060058594
detect face (primed): 0.002499818801879883
smooth + extract face (primed): 4.4345855712890625e-05
detect landmark (primed): 0.0008718967437744141
smooth landmark (primed): 0.0005543231964111328
detect gaze (primed): 0.003607034683227539
smooth gaze (primed): 1.1444091796875e-05
visualize gaze: 0.0006237030029296875
create plots: 5.7220458984375e-06
get gaze mask: 0.0002582073211669922
prep yolo img: 0.001455068588256836
yolo pred: 0.0012624263763427734
total yolo: 0.0027174949645996094
draw and get yolo boxes: 0.003875732421875
segment one mask: 0.0046160221099853516

display image: 0.0020477771759033203
save to file (out/quantized_yolo/1700710189.1127303.png): 1.8956935405731201
non-load total: 0.05884432792663574
load total: 0.7904863357543945

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
    
        
        
        
        
        
        
        
        