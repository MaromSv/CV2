# This is a conceptual example - you'll need to adapt to your actual dataloader
class Multi_GoPro_Loader(Dataset):
    def __getitem__(self, idx):
        # ... existing code to load blur and sharp images ...
        
        # Load or compute blur field ground truth (dx, dy, magnitude)
        blur_field_path = os.path.join(self.blur_field_dir, self.image_list[idx])
        blur_field = self.load_blur_field(blur_field_path)
        
        # Return both the blur image and the blur field
        return {
            'blur': blur,
            'blur_field': blur_field,  # 3-channel tensor [dx, dy, magnitude]
            'sharp': sharp  # Optional, if you still need it
        }