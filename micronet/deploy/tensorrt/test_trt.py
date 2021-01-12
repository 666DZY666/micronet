# test_trt(temp_demo)

import numpy as np
import torch
import torch.nn as nn
# tensorrt-lib
import util_trt
from calibrator import SegBatchStream

# trt test function
def test_trt(segmentation_module_trt, loader, gpu):
    infer_time_sum = 0.0
    warmup_flag = 0
    warmup_nums = 5
    #pbar = tqdm(total=len(loader))

    for batch_data in loader:
        warmup_flag += 1
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']
        resized_nums = len(img_resized_list)

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp, infer_time = segmentation_module_trt(feed_dict, segSize=segSize, shape_of_input=img.shape)
                if warmup_flag > warmup_nums:
                    infer_time_sum = infer_time_sum + infer_time
                scores = scores + pred_tmp.cuda() / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

            # import pdb
            # pdb.set_trace()
            # cv2.imwrite('sl.jpg', (pred*255).astype(np.uint8))

        # Judge whether the image is saliency 
        if np.mean(scores.squeeze(0)[1,:,:].squeeze(0).cpu().numpy()) < 0.01 or np.mean(pred) < 0.001:
            pred = as_numpy(torch.zeros(segSize[0], segSize[1]))
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

        # Connection Region Judge
        saliecy_map = scores.squeeze(0)[1,:,:].squeeze(0).cpu().numpy()
        saliecy_map *= 255
        ret,thresh=cv2.threshold(saliecy_map.astype(np.uint8),127,255,0)
        contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours), batch_data['info'])
        # if len(contours) > 5
        valid_region = 0
        max_region = 0
        for idx, contour in enumerate(contours):
            x1, y1 = np.min(contour[:,:,0]), np.min(contour[:,:,1])
            x2, y2 = np.max(contour[:,:,0]), np.max(contour[:,:,1])
            if hierarchy[0][idx][-1] != -1:   # Children Region
                continue
            valid_region += 1
            area = (x2 - x1) * (y2 - y1)    
            if area < 300:    # Area Filter
                pred[y1:y2,x1:x2] = 0
                continue 
            if area > max_region:
                max_region = area 
                
        #print(valid_region, len(contours), batch_data['info'])
            
        # 黑色背景
        visualize_result_black_trt(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg,
            scores.squeeze(0)[1,:,:].squeeze(0).cpu().numpy()
        )
        #pbar.update(1)
    fps = (len(loader) - warmup_nums) * resized_nums / infer_time_sum
    return fps

def main(cfg, gpu):
    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    # model
    net_resnet = build_resnet_upsample(
        arch=cfg.MODEL.arch_encoder.lower(),
        arch_de=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        fc_dim_de=cfg.MODEL.fc_dim,
        num_class_de=cfg.DATASET.num_class,
        use_softmax=True,
        deep_sup_scale=None,
        segSize=None)
    net_resnet.load_state_dict(torch.load(cfg.MODEL.weights_seg, map_location=lambda storage, loc: storage), strict=True)

    # torch2onnx 
    input_name = ['input']
    output_name = ['output']
    onnx_model_fixed = 'models_save/model_seg_fixed.onnx'
    onnx_model_dynamic = 'models_save/model_seg_dynamic.onnx'
    batch_size = 1
    img_size_fixed = (3, 400, 400)
    img_size_dynamic = (3, 800, 800)
    dummy_input_fixed = torch.rand(batch_size, *img_size_fixed)
    dummy_input_dynamic = torch.rand(batch_size, *img_size_dynamic)
    dynamic_axes = {'input': {2: "height", 3: "width"}}
    print('\n*** torch to onnx begin ***')
      # fixed_onnx
    torch.onnx.export(net_resnet, dummy_input_fixed, onnx_model_fixed, verbose=True, 
        input_names=input_name, output_names=output_name, opset_version=10)
      # dynamic_onnx
    torch.onnx.export(net_resnet, dummy_input_dynamic, onnx_model_dynamic, verbose=True, 
        input_names=input_name, output_names=output_name, opset_version=10, dynamic_axes=dynamic_axes)
    print('*** torch to onnx completed ***\n')
    # onnx2trt
    fp16_mode = False
    int8_mode = True 
    transform = None
    print('*** onnx to tensorrt begin ***')
    max_calibration_size = 80  # 校准集数量
    calibration_batche_size = 10  # 校准batch_size
    max_calibration_batches = max_calibration_size / calibration_batche_size
      # calibration
    calibration_stream = SegBatchStream(dataset_test, transform, calibration_batche_size, img_size_fixed, max_batches=max_calibration_batches)
    engine_model_fixed = "models_save/model_seg_fixed.trt"
    engine_model_dynamic = "models_save/model_seg_dynamic.trt"
    calibration_table = 'models_save/calibration_seg.cache'
      # fixed_engine,校准产生校准表
    engine_fixed = util_trt.get_engine(batch_size, onnx_model_fixed, engine_model_fixed, fp16_mode=fp16_mode, 
        int8_mode=int8_mode, calibration_stream=calibration_stream, calibration_table_path=calibration_table, save_engine=True, dynamic=False)
    assert engine_fixed, 'Broken engine_fixed'
    print('*** engine_fixed completed ***\n')
      # dynamic_engine,加载fixed_engine生成的校准表,用于inference
    engine_dynamic = util_trt.get_engine(batch_size, onnx_model_dynamic, engine_model_dynamic, fp16_mode=fp16_mode, 
        int8_mode=int8_mode, calibration_stream=calibration_stream, calibration_table_path=calibration_table, save_engine=True, dynamic=True)
    assert engine_dynamic, 'Broken engine_dynamic'
    print('*** engine_dynamic completed ***\n')
    print('*** onnx to tensorrt completed ***\n')
      # context and buffer
    context = engine_dynamic.create_execution_context() 
      # choose an optimization profile
    context.active_optimization_profile = 0
    buffers = util_trt.allocate_buffers_v2(engine_dynamic, 1200, 1200)
    
    # trt test
    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module_trt = SegmentationModule_v2_trt(context, buffers, crit, use_softmax=True, binding_id=0)
    segmentation_module_trt.cuda()
    # result
    print('*** trt_model ***')
    print('test ing...')
    fps_trt = test_trt(segmentation_module_trt, loader_test, gpu)
    print('trt_model test done!')
    '''
    print('trt-fps: ', fps_trt)
    '''

if __name__ == '__main__':
    main(cfg, args.gpu)
