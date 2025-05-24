import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
from watermark_utils import apply_pca_to_watermark, embed_watermark_lsb, extract_watermark_lsb, verify_watermark

def main():
    st.title("Aplikasi Watermarking Digital dengan PCA-LSB")
    
    # CSS untuk styling tombol mode
    st.markdown("""
        <style>
        .stRadio [role=radiogroup] {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Pilihan mode aplikasi di bagian atas
    st.write("### Pilih Mode Aplikasi")
    mode = st.radio(
        "",  # Label kosong karena sudah ada judul di atas
        ["Tambah Watermark", "Verifikasi Watermark"],
        horizontal=True  # Membuat pilihan radio horizontal
    )
    
    # Garis pemisah
    st.markdown("---")
    
    if mode == "Tambah Watermark":
        st.write("""
        Aplikasi ini menggunakan kombinasi teknik PCA (Principal Component Analysis) dan 
        LSB (Least Significant Bit) untuk melakukan watermarking pada gambar digital.
        """)

        # Informasi persyaratan
        st.info("""
        ℹ️ Persyaratan Gambar:
        1. Format yang didukung: JPG, JPEG, PNG
        2. Gambar Watermark:
           - Ukuran lebih kecil dari gambar utama
           - Memiliki kontras yang baik
           - Bisa berwarna atau hitam putih
        3. Gambar Utama:
           - Ukuran harus cukup besar
           - Sebaiknya memiliki variasi warna yang tidak terlalu ekstrem
        """)

        # Upload gambar utama
        st.header("1. Upload Gambar Utama")
        cover_image_file = st.file_uploader("Pilih gambar utama (cover image)", type=['jpg', 'jpeg', 'png'], key='cover')
        
        # Upload watermark
        st.header("2. Upload Watermark")
        watermark_file = st.file_uploader("Pilih gambar watermark", type=['jpg', 'jpeg', 'png'], key='watermark')

        if cover_image_file is not None and watermark_file is not None:
            # Tampilkan gambar yang diupload
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gambar Utama")
                cover_image = Image.open(cover_image_file)
                st.image(cover_image, use_container_width=True)
                st.write(f"Ukuran: {cover_image.size[0]} x {cover_image.size[1]} pixel")
                
            with col2:
                st.subheader("Watermark")
                watermark_image = Image.open(watermark_file)
                st.image(watermark_image, use_container_width=True)
                st.write(f"Ukuran: {watermark_image.size[0]} x {watermark_image.size[1]} pixel")

            # Periksa ukuran watermark
            if watermark_image.size[0] * watermark_image.size[1] > cover_image.size[0] * cover_image.size[1]:
                st.warning("⚠️ Peringatan: Ukuran watermark lebih besar dari gambar utama. Ini mungkin menyebabkan hasil yang tidak optimal.")

            # Tombol untuk memproses watermark
            if st.button("Proses Watermarking", type="primary"):  # Menambahkan style primary
                with st.spinner("Sedang memproses..."):
                    try:
                        # Konversi gambar ke array numpy
                        cover_array = np.array(cover_image)
                        watermark_array = np.array(watermark_image)

                        # Terapkan PCA pada watermark
                        watermark_pca, _, _ = apply_pca_to_watermark(watermark_array)

                        # Sisipkan watermark menggunakan LSB
                        watermarked_image = embed_watermark_lsb(cover_array, watermark_pca)

                        # Tampilkan hasil
                        st.header("3. Hasil Watermarking")
                        st.image(watermarked_image, caption="Gambar dengan Watermark", use_container_width=True)

                        # Konversi hasil ke bytes untuk download
                        result_image = Image.fromarray(watermarked_image.astype('uint8'))
                        buf = io.BytesIO()
                        result_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()

                        # Tombol download dengan style
                        st.download_button(
                            label="Download Hasil",
                            data=byte_im,
                            file_name="watermarked_image.png",
                            mime="image/png",
                            type="primary"
                        )

                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")

    else:  # Mode Verifikasi Watermark
        st.write("""
        ## Verifikasi Watermark
        Upload gambar untuk memeriksa apakah terdapat watermark di dalamnya.
        """)
        
        # Upload gambar untuk verifikasi
        verify_image_file = st.file_uploader("Upload gambar untuk verifikasi", type=['jpg', 'jpeg', 'png'], key='verify')
        
        if verify_image_file is not None:
            verify_image = Image.open(verify_image_file)
            st.image(verify_image, caption="Gambar yang akan diverifikasi", use_container_width=True)
            
            if st.button("Periksa Watermark", type="primary"):  # Menambahkan style primary
                with st.spinner("Sedang memeriksa watermark..."):
                    try:
                        # Konversi gambar ke array numpy
                        verify_array = np.array(verify_image)
                        
                        # Ekstrak dan periksa watermark
                        has_watermark, confidence, description = verify_watermark(verify_array)
                        
                        # Tampilkan hasil verifikasi
                        if has_watermark:
                            st.success(description)
                        else:
                            st.warning(description)
                            
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat verifikasi: {str(e)}")

if __name__ == "__main__":
    main() 