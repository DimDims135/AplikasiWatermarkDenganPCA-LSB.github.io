import numpy as np
from sklearn.decomposition import PCA
import cv2

def apply_pca_to_watermark(watermark_image, n_components=None):
    """
    Menerapkan PCA pada gambar watermark
    """
    # Konversi gambar ke grayscale jika belum
    if len(watermark_image.shape) > 2:
        watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)
    else:
        watermark_gray = watermark_image.copy()
    
    # Pastikan gambar dalam format uint8
    watermark_gray = watermark_gray.astype(np.uint8)
    
    # Normalisasi gambar ke range [0, 1]
    watermark_normalized = watermark_gray.astype(float) / 255.0
    
    # Reshape gambar ke matrix 2D
    height, width = watermark_normalized.shape
    X = watermark_normalized.reshape(height, -1)
    
    # Hitung jumlah komponen
    if n_components is None:
        n_components = 1
    
    # Terapkan PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Normalisasi hasil PCA ke range [0, 1]
    X_pca = (X_pca - X_pca.min()) / (X_pca.max() - X_pca.min() + 1e-10)
    
    # Pastikan nilai dalam range [0, 1]
    X_pca = np.clip(X_pca, 0, 1)
    
    return X_pca.reshape(-1, 1), pca, (height, width)

def embed_watermark_lsb(cover_image, watermark_data):
    """
    Menyisipkan data watermark ke dalam LSB cover image dengan metode yang lebih robust
    """
    # Buat salinan gambar cover
    cover_array = np.array(cover_image).copy()
    
    # Pastikan cover image dalam format uint8
    if cover_array.dtype != np.uint8:
        cover_array = cover_array.astype(np.uint8)
    
    # Flatten dan normalisasi watermark
    watermark_flat = watermark_data.flatten()
    watermark_flat = np.clip(watermark_flat, 0, 1)
    
    # Konversi ke nilai 8-bit
    watermark_8bit = (watermark_flat * 255).astype(np.uint8)
    
    # Buat array untuk menyimpan hasil
    result = cover_array.copy()
    
    # Jika gambar berwarna, proses setiap channel
    if len(result.shape) == 3:
        height, width, channels = result.shape
        for c in range(channels):
            channel = result[:, :, c]
            channel_flat = channel.flatten()
            
            # Sisipkan watermark dengan pola yang lebih kuat
            for i in range(len(watermark_8bit)):
                if i >= len(channel_flat):
                    break
                    
                # Ambil 8 bit dari watermark
                watermark_byte = watermark_8bit[i]
                
                # Sisipkan di multiple posisi untuk redundansi
                positions = [
                    i,  # Posisi utama
                    (i + width) % len(channel_flat),  # Posisi offset vertikal
                    (i * 2) % len(channel_flat)  # Posisi offset diagonal
                ]
                
                for pos in positions:
                    # Modifikasi 2 bit terakhir untuk watermark yang lebih kuat
                    channel_flat[pos] = (channel_flat[pos] & 0xFC) | (watermark_byte & 0x03)
            
            # Reshape kembali channel
            result[:, :, c] = channel_flat.reshape((height, width))
    else:
        # Untuk gambar grayscale
        height, width = result.shape
        result_flat = result.flatten()
        
        for i in range(len(watermark_8bit)):
            if i >= len(result_flat):
                break
                
            watermark_byte = watermark_8bit[i]
            positions = [
                i,
                (i + width) % len(result_flat),
                (i * 2) % len(result_flat)
            ]
            
            for pos in positions:
                result_flat[pos] = (result_flat[pos] & 0xFC) | (watermark_byte & 0x03)
        
        result = result_flat.reshape((height, width))
    
    # Tambahkan marker khusus di beberapa lokasi untuk memudahkan deteksi
    marker = np.array([0xAA, 0x55, 0xAA, 0x55], dtype=np.uint8)  # Pola marker
    if len(result.shape) == 3:
        for c in range(channels):
            result[0:4, 0, c] = marker  # Marker di pojok kiri atas
            result[0:4, -1, c] = marker  # Marker di pojok kanan atas
    else:
        result[0:4, 0] = marker  # Marker di pojok kiri atas
        result[0:4, -1] = marker  # Marker di pojok kanan atas
    
    return result

def extract_watermark_lsb(watermarked_image, original_shape, n_components):
    """
    Mengekstrak watermark dari LSB watermarked image dengan metode voting
    """
    # Flatten image
    watermarked_flat = watermarked_image.flatten()
    
    # Ekstrak LSB dengan voting
    extracted_bits = []
    max_bytes = n_components * 8
    
    for i in range(max_bytes):
        votes = []
        # Ambil bit dari beberapa posisi
        positions = [i]
        if i + max_bytes < len(watermarked_flat):
            positions.append(i + max_bytes)
        if i + 2*max_bytes < len(watermarked_flat):
            positions.append(i + 2*max_bytes)
        
        # Voting untuk menentukan bit yang benar
        for pos in positions:
            # Ambil 2 bit terakhir
            votes.extend([(watermarked_flat[pos] & 2) >> 1, watermarked_flat[pos] & 1])
        
        # Tentukan bit berdasarkan mayoritas
        extracted_bits.append(1 if sum(votes) > len(votes)/2 else 0)
    
    # Konversi bits ke bytes
    extracted_bytes = []
    for i in range(0, len(extracted_bits), 8):
        if i + 8 <= len(extracted_bits):
            byte = ''.join(str(b) for b in extracted_bits[i:i+8])
            extracted_bytes.append(int(byte, 2))
    
    # Konversi ke array dan normalisasi
    extracted_data = np.array(extracted_bytes, dtype=np.uint8) / 255.0
    
    return extracted_data

def verify_watermark(image):
    """
    Memverifikasi keberadaan watermark dalam gambar dan memberikan deskripsi detail hasilnya
    Returns:
        - bool: True jika watermark terdeteksi
        - float: Tingkat keyakinan dalam persen
        - str: Deskripsi detail hasil verifikasi
    """
    try:
        # Konversi ke array numpy
        img_array = np.array(image)
        
        # Cek marker terlebih dahulu
        marker = np.array([0xAA, 0x55, 0xAA, 0x55], dtype=np.uint8)
        has_marker = False
        marker_locations = []
        
        if len(img_array.shape) == 3:
            # Cek marker di setiap channel
            for c in range(img_array.shape[2]):
                if np.array_equal(img_array[0:4, 0, c] & 0x03, marker & 0x03):
                    has_marker = True
                    marker_locations.append("pojok kiri atas")
                if np.array_equal(img_array[0:4, -1, c] & 0x03, marker & 0x03):
                    has_marker = True
                    marker_locations.append("pojok kanan atas")
        else:
            # Cek marker untuk gambar grayscale
            if np.array_equal(img_array[0:4, 0] & 0x03, marker & 0x03):
                has_marker = True
                marker_locations.append("pojok kiri atas")
            if np.array_equal(img_array[0:4, -1] & 0x03, marker & 0x03):
                has_marker = True
                marker_locations.append("pojok kanan atas")
        
        # Siapkan deskripsi detail
        description = ""
        
        # Jika marker ditemukan, lakukan analisis lebih detail
        if has_marker:
            description += f"✓ Marker watermark terdeteksi di {', '.join(marker_locations)}.\n"
            
            # Analisis pola LSB
            if len(img_array.shape) == 3:
                channel = img_array[:, :, 0]  # Analisis channel pertama
                description += "✓ Menganalisis channel warna pertama untuk pola watermark.\n"
            else:
                channel = img_array
                description += "✓ Menganalisis gambar grayscale untuk pola watermark.\n"
            
            flat_image = channel.flatten()
            lsb_pattern = flat_image & 0x03  # Ambil 2 bit terakhir
            
            # Hitung statistik pola
            unique_patterns, pattern_counts = np.unique(lsb_pattern, return_counts=True)
            pattern_distribution = pattern_counts / len(lsb_pattern)
            
            # Hitung entropy dari distribusi pola
            entropy = -np.sum(pattern_distribution * np.log2(pattern_distribution + 1e-10))
            
            # Watermark biasanya memiliki entropy yang moderat
            confidence = 100 * (1 - abs(entropy - 1.5) / 1.5)
            confidence = np.clip(confidence, 0, 100)
            
            # Tambahkan detail analisis
            if confidence > 75:
                description += f"✓ Pola watermark sangat jelas terdeteksi (Entropy: {entropy:.2f}).\n"
            elif confidence > 50:
                description += f"✓ Pola watermark terdeteksi dengan cukup jelas (Entropy: {entropy:.2f}).\n"
            elif confidence > 25:
                description += f"⚠ Pola watermark terdeteksi tetapi kurang jelas (Entropy: {entropy:.2f}).\n"
            else:
                description += f"⚠ Pola watermark sangat lemah (Entropy: {entropy:.2f}).\n"
            
            # Kesimpulan
            description += f"\nKesimpulan: Gambar ini MEMILIKI watermark dengan tingkat keyakinan {confidence:.1f}%."
            return True, confidence, description
        else:
            description = "✗ Tidak ditemukan marker watermark di gambar.\n"
            description += "✗ Tidak ada pola watermark yang terdeteksi.\n"
            description += "\nKesimpulan: Gambar ini TIDAK MEMILIKI watermark yang valid."
            return False, 0.0, description
        
    except Exception as e:
        error_description = f"Error saat melakukan verifikasi: {str(e)}\n"
        error_description += "\nKesimpulan: Tidak dapat memverifikasi watermark karena terjadi error."
        return False, 0.0, error_description 