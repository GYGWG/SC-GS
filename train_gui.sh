# train static scene
# CUDA_VISIBLE_DEVICES=1 python train_gui.py \
#     --source_path "data/campsite-small" \
#     --model_path "outputs/campsite/" \
#     --is_scene_static \
#     --deform_type "node" \
#     --node_num "512" \
#     --gt_alpha_mask_as_dynamic_mask \
#     --gs_with_motion_mask \
#     --W "800" \
#     --H "800" \
#     --white_background \
#     --init_isotropic_gs_with_all_colmap_pcl

# train dynamic scene
CUDA_VISIBLE_DEVICES=0 python train_gui.py \
     --source_path data/bouncingballs \
     --model_path outputs/bouncingballs_test \
     --label_folder output_mask_colored\
     --deform_type node \
     --node_num 512 \
     --hyper_dim 8 \
     --is_blender \
     --eval \
     --gt_alpha_mask_as_scene_mask \
     --local_frame \
     --resolution 2 \
     --gui \
     --W 800 \
     --H 800