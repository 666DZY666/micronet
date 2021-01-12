# eval_trt(temp_demo)

import numpy as np
import torch
import torch.nn as nn
# tensorrt-lib
import util_trt
from calibrator import SegBatchStream

# trt eval function
def evaluate_trt(segmentation_module_trt, loader, cfg, gpu, result_queue_trt):
    #pbar = tqdm(total=len(loader))

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                scores_tmp, infer_time = segmentation_module_trt(feed_dict, segSize=segSize, shape_of_input=img.shape)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            # print(scores.squeeze(0)[1,:,:].squeeze(0).shape, pred.shape)
            pred = as_numpy(pred.squeeze(0).cpu())

        # calculate accuracy and SEND THEM TO MASTER
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        result_queue_trt.put_nowait((acc, pix, intersection, union))

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result'),
                scores.squeeze(0)[1,:,:].squeeze(0).cpu().numpy()
            )
        #pbar.update(1)

def main(cfg, gpu):
    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

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
    max_calibration_size = 100    # 校准集数量
    calibration_batche_size = 16  # 校准batch_size
    max_calibration_batches = max_calibration_size / calibration_batche_size
      # calibration
    calibration_stream = SegBatchStream(dataset_val, transform, calibration_batche_size, img_size_fixed, max_batches=max_calibration_batches)
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
    
    # trt eval
    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module_trt = SegmentationModule_v2_trt(context, buffers, crit, use_softmax=True, binding_id=0)
    segmentation_module_trt.cuda()
    print('*** trt_model ***')
    print('eval ing...')
    evaluate_trt(segmentation_module_trt, loader_val, cfg, gpu, result_queue_trt)
    print('trt_model eval done!')
    # result
    print('\r\nresult')
    print('*** trt_model ***')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%'
          .format(iou_trt.mean(), acc_meter_trt.average()*100))
    
if __name__ == '__main__':
    main(cfg, args.gpu)
    