import tensorflow as tf
import cv2
import numpy as np
import os
from collections import deque
import traceback
import threading
import math
import socket 
import json  
import time  

try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("UYARI: 'scikit-image' kütüphanesi bulunamadı. İskelet çıkarma bazı durumlarda çalışmayabilir.")
    SKIMAGE_AVAILABLE = False
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

MODEL_PATH = 'final_maze_segmentation_unet_model.h5'  
THRESHOLD = 0.5
SMALL_STEP_THRESHOLD = 6 
MIN_ACCEPTABLE_FORWARD_STEP = 10 
LIVE_PATH_LINE_COLOR = (0, 0, 255)  
LIVE_PATH_LINE_THICKNESS = 3
LIVE_START_POINT_COLOR = (0, 255, 0) 
LIVE_END_POINT_COLOR = (255, 0, 0)  
LIVE_POINT_RADIUS = 9
LIVE_POINT_THICKNESS = -1 
COMMAND_PATH_LINE_COLOR = (255, 100, 0) 
COMMAND_PATH_LINE_THICKNESS = 4
TURN_MARKER_COLOR = (0, 255, 255)  
TURN_MARKER_RADIUS = 7
TURN_MARKER_THICKNESS = -1  
VEHICLE_IMAGE_FILENAME = "arac.png"  
ANIMATION_DELAY_MS = 100
STEPS_PER_SEGMENT = 20 
PIXEL_MOVE_PER_COMMAND_ANIM_STEP = 10  


SERVER_HOST = '0.0.0.0' 
SERVER_PORT = 65432  


class MazeSolverApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Labirent Çözücü ve Raspberry Pi Kontrol Arayüzü")
        self.root.geometry("1350x900")

        self.model = None
        self.input_shape_tuple = None
        self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS = None, None, None
        self.is_grayscale_model = False

        self.image_path = None
        self.original_cv_image = None
        self.h_orig_for_path, self.w_orig_for_path = 0, 0
        self.image_with_skeleton_overlay_for_selection = None
        self.display_image_tk = None
        self.displayed_image_pil = None
        self.current_image_source = None 

        self.mask_for_bfs_and_clicking_ORIG_SCALE = None
        self.padding_info = {}
        self.start_point_original_coords = None
        self.end_point_original_coords = None
        self.selected_points_on_canvas = []

        self.current_mask_type_str = "Bilinmeyen"
        self.raw_model_mask_pil = None
        self.skeleton_model_scale_pil = None
        self.bfs_mask_pil = None

        self.result_image_display_tk_pil = None
        self.live_camera_feed_tk = None

        self.last_simplified_path_for_overlay = None
        self.last_generated_commands_for_pi_json = []
        self.last_generated_commands_for_display = []


        self.camera_thread = None
        self.stop_camera_event = threading.Event()
        self.is_live_camera_active = False
        self.video_capture_device = None

        self.is_camera_streaming_on_main_canvas = False
        self.camera_preview_thread = None
        self.stop_camera_preview_event = threading.Event()
        self.video_capture_device_main_canvas = None
        self.current_preview_frame_cv2 = None

        self.original_vehicle_pil = None
        self.is_path_animating = False
        self.path_animation_job_id = None
        self.current_coord_path_segment_index = 0
        self.current_step_in_coord_segment = 0

        self.is_command_animating = False
        self.command_animation_job_id = None
        self.current_command_index_for_anim = 0 
        self.vehicle_pos_x_cmd_anim = 0.0
        self.vehicle_pos_y_cmd_anim = 0.0
        self.vehicle_orientation_dy_anim = 0 
        self.vehicle_orientation_dx_anim = 0 
        self.steps_taken_for_current_fwd_cmd_anim = 0
        self.total_steps_for_current_fwd_cmd_anim = 0
        self.current_animation_frame_pil = None


        self.tab_skeleton_mask = None
        self.tab_raw_model_mask = None
        self.lbl_skeleton_mask_canvas = None
        self.lbl_raw_model_mask_canvas = None

        self.socket_server_thread = None
        self.stop_socket_server_event = threading.Event()
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.is_server_running = False
        self.server_status_message = "Sunucu: Başlatılmadı"
        self.pi_status_message = "Raspberry Pi: Bağlı Değil"
        self.is_pi_calibrating = False
        self.is_pi_driving = False
        self.pi_calibration_offset = None

        self.setup_ui()
        self.load_model_on_startup()
        self.load_vehicle_image()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        self.control_frame = ttk.LabelFrame(self.root, text="Kontroller", padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.btn_load_gallery = ttk.Button(self.control_frame, text="Galeriden Görüntü Seç",
                                           command=self.load_image_from_gallery)
        self.btn_load_gallery.pack(pady=5, fill=tk.X)

        self.btn_load_camera = ttk.Button(self.control_frame, text="Kameradan Görüntü Al",
                                          command=self.toggle_camera_stream_and_capture)
        self.btn_load_camera.pack(pady=5, fill=tk.X)

        self.lbl_image_status = ttk.Label(self.control_frame, text="Görüntü seçilmedi.")
        self.lbl_image_status.pack(pady=5, fill=tk.X)

        self.lbl_point_instruction = ttk.Label(self.control_frame, text="1. Başlangıç noktasını seçin.")
        self.lbl_point_instruction.pack(pady=10, fill=tk.X)

        self.btn_process = ttk.Button(self.control_frame, text="Yolu Bul ve İşle", command=self.process_maze,
                                      state=tk.DISABLED)
        self.btn_process.pack(pady=10, fill=tk.X)

        self.animation_control_frame = ttk.LabelFrame(self.control_frame, text="Animasyon (PC)", padding=5)
        self.animation_control_frame.pack(pady=10, fill=tk.X)

        self.btn_animate_path = ttk.Button(self.animation_control_frame, text="Yol Koord. Anime Et",
                                           command=self.start_path_animation, state=tk.DISABLED)
        self.btn_animate_path.pack(pady=5, fill=tk.X)

        self.btn_animate_commands = ttk.Button(self.animation_control_frame, text="Komutlarla Anime Et",
                                               command=self.start_command_animation, state=tk.DISABLED)
        self.btn_animate_commands.pack(pady=5, fill=tk.X)

        self.rpi_control_frame = ttk.LabelFrame(self.control_frame, text="Raspberry Pi Kontrolü", padding=5)
        self.rpi_control_frame.pack(pady=10, fill=tk.X)

        self.btn_drive_vehicle = ttk.Button(self.rpi_control_frame, text="Aracı Sür (Pi)",
                                            command=self.drive_vehicle_on_pi, state=tk.DISABLED)
        self.btn_drive_vehicle.pack(pady=5, fill=tk.X)

        self.btn_stop_vehicle_on_pi = ttk.Button(self.rpi_control_frame, text="Aracı Durdur (Pi)",
                                                 command=self.stop_vehicle_on_pi, state=tk.DISABLED)
        self.btn_stop_vehicle_on_pi.pack(pady=5, fill=tk.X)

        self.lbl_pi_status = ttk.Label(self.rpi_control_frame, text=self.pi_status_message)
        self.lbl_pi_status.pack(pady=5, fill=tk.X)

        self.server_control_frame = ttk.LabelFrame(self.control_frame, text="Ağ Sunucusu (PC)", padding=5)
        self.server_control_frame.pack(pady=10, fill=tk.X, side=tk.BOTTOM)

        self.btn_toggle_server = ttk.Button(self.server_control_frame, text="Sunucuyu Başlat",
                                            command=self.toggle_socket_server)
        self.btn_toggle_server.pack(pady=5, fill=tk.X)

        self.lbl_server_status = ttk.Label(self.server_control_frame, text=self.server_status_message)
        self.lbl_server_status.pack(pady=5, fill=tk.X)

        self.btn_reset = ttk.Button(self.control_frame, text="Sıfırla (Tüm Sistem)", command=self.reset_all_app_state)
        self.btn_reset.pack(pady=20, fill=tk.X, side=tk.BOTTOM)

        self.progress_bar = ttk.Progressbar(self.control_frame, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress_bar.pack(pady=10, side=tk.BOTTOM)

        self.image_interaction_frame = ttk.LabelFrame(self.root, text="Labirent Görüntüsü (Yol Seçimi)", padding=10)
        self.image_interaction_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_image = tk.Canvas(self.image_interaction_frame, bg="gray")
        self.canvas_image.pack(fill=tk.BOTH, expand=True)
        self.canvas_image.bind("<Button-1>", self.on_image_click)
        self.canvas_image.bind("<Configure>", self.on_main_canvas_resize)

        self.results_frame = ttk.LabelFrame(self.root, text="Sonuçlar", padding=10)
        self.results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.notebook_results = ttk.Notebook(self.results_frame)
        self.notebook_results.bind("<<NotebookTabChanged>>", self.on_results_tab_changed)

        self.tab_result_image = ttk.Frame(self.notebook_results)
        self.tab_skeleton_mask = ttk.Frame(self.notebook_results)
        self.tab_raw_model_mask = ttk.Frame(self.notebook_results)
        self.tab_commands = ttk.Frame(self.notebook_results)
        self.tab_live_camera = ttk.Frame(self.notebook_results)

        self.notebook_results.add(self.tab_result_image, text='Çözülmüş Labirent')
        self.notebook_results.add(self.tab_skeleton_mask, text='İskelet Maskesi')
        self.notebook_results.add(self.tab_raw_model_mask, text='Modelin Ham Maskesi')
        self.notebook_results.add(self.tab_commands, text='Komutlar (Pi için)')
        self.notebook_results.add(self.tab_live_camera, text='Canlı Kamera & Yol')
        self.notebook_results.pack(expand=1, fill='both')

        self.lbl_result_image_canvas = tk.Canvas(self.tab_result_image, bg="lightgrey")
        self.lbl_result_image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.lbl_result_image_canvas.bind("<Configure>", lambda e, c=self.lbl_result_image_canvas,
                                           i_attr='result_image_display_tk_pil': self.on_generic_canvas_resize_wrapper(
            e, c, i_attr, 'current_animation_frame_pil'))

        self.skeleton_display_frame = ttk.Frame(self.tab_skeleton_mask)
        self.skeleton_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.lbl_skeleton_title = ttk.Label(self.skeleton_display_frame, text="İskelet Maskesi (Model Ölçeği)")
        self.lbl_skeleton_title.pack(pady=(0, 5))
        self.lbl_skeleton_mask_canvas = tk.Canvas(self.skeleton_display_frame, bg="lightgrey")
        self.lbl_skeleton_mask_canvas.pack(fill=tk.BOTH, expand=True)
        self.lbl_skeleton_mask_canvas.bind("<Configure>", lambda e, c=self.lbl_skeleton_mask_canvas,
                                           i_attr='skeleton_model_scale_pil': self.on_generic_canvas_resize(
            e, c, i_attr))

        self.raw_model_display_frame = ttk.Frame(self.tab_raw_model_mask)
        self.raw_model_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.lbl_raw_model_title = ttk.Label(self.raw_model_display_frame, text="Modelin Ham Maskesi (Model Ölçeği)")
        self.lbl_raw_model_title.pack(pady=(0, 5))
        self.lbl_raw_model_mask_canvas = tk.Canvas(self.raw_model_display_frame, bg="lightgrey")
        self.lbl_raw_model_mask_canvas.pack(fill=tk.BOTH, expand=True)
        self.lbl_raw_model_mask_canvas.bind("<Configure>", lambda e, c=self.lbl_raw_model_mask_canvas,
                                            i_attr='raw_model_mask_pil': self.on_generic_canvas_resize(
            e, c, i_attr))

        self.txt_commands = tk.Text(self.tab_commands, wrap=tk.WORD, height=10, width=40)
        self.scroll_commands = ttk.Scrollbar(self.tab_commands, command=self.txt_commands.yview)
        self.txt_commands.config(yscrollcommand=self.scroll_commands.set)
        self.scroll_commands.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_commands.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.live_camera_canvas = tk.Canvas(self.tab_live_camera, bg="black")
        self.live_camera_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.live_camera_status_label = ttk.Label(self.tab_live_camera, text="Yol bulunursa kamera burada aktifleşir.")
        self.live_camera_status_label.pack(pady=5)

    def _update_server_status_ui(self, message):
        self.server_status_message = message
        if hasattr(self, 'lbl_server_status') and self.lbl_server_status.winfo_exists():
            self.root.after(0, lambda: self.lbl_server_status.config(text=self.server_status_message))
        print(message)

    def _update_pi_status_ui(self, message):
        self.pi_status_message = message
        if hasattr(self, 'lbl_pi_status') and self.lbl_pi_status.winfo_exists():
            self.root.after(0, lambda: self.lbl_pi_status.config(text=self.pi_status_message))
        print(f"PI_STATUS: {message}")

    def toggle_socket_server(self):
        if self.is_server_running:
            self._update_server_status_ui("Sunucu: Durduruluyor...")
            self.stop_socket_server_event.set()
            if self.client_socket:
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                    self.client_socket.close()
                except:
                    pass
                self.client_socket = None
                self._update_pi_status_ui("Raspberry Pi: Bağlantı Kesildi (Sunucu Durdu)")
                self.btn_drive_vehicle.config(state=tk.DISABLED)
                self.btn_stop_vehicle_on_pi.config(state=tk.DISABLED)

            if self.socket_server_thread and self.socket_server_thread.is_alive():
                try:
                    if self.server_socket:
                        self.server_socket.close() 
                    self.socket_server_thread.join(timeout=2.0)
                except Exception as e:
                    print(f"Sunucu thread'i join hatası: {e}")

            self.is_server_running = False
            self.btn_toggle_server.config(text="Sunucuyu Başlat")
            self._update_server_status_ui("Sunucu: Durduruldu")
            self.server_socket = None 
        else:
            self.is_server_running = True
            self.stop_socket_server_event.clear()
            self.btn_toggle_server.config(text="Sunucuyu Durdur")
            self._update_server_status_ui(f"Sunucu: {SERVER_HOST}:{SERVER_PORT} üzerinde başlatılıyor...")

            self.socket_server_thread = threading.Thread(target=self._socket_server_loop, daemon=True)
            self.socket_server_thread.start()

    def _socket_server_loop(self):
        if self.server_socket is None: 
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.server_socket.bind((SERVER_HOST, SERVER_PORT))
                self.server_socket.listen(1)
                self.root.after(0, lambda: self._update_server_status_ui(
                    f"Sunucu: {SERVER_HOST}:{SERVER_PORT} dinleniyor..."))
            except socket.error as e:
                self.root.after(0, lambda msg=f"Sunucu başlatma hatası: {e}": self._update_server_status_ui(msg))
                self.root.after(0, self.toggle_socket_server) 
                self.is_server_running = False 
                if self.server_socket: self.server_socket.close() 
                self.server_socket = None 
                return 

        while self.is_server_running and not self.stop_socket_server_event.is_set():
            try:
                if not self.server_socket: 
                    print("Sunucu soketi None, döngüden çıkılıyor.")
                    break

                self.server_socket.settimeout(1.0) 
                conn, addr = self.server_socket.accept()

                if self.client_socket is not None:
                    print(f"Zaten bir istemci ({self.client_address}) bağlı. Yeni bağlantı ({addr}) reddedildi.")
                    conn.sendall(b"BUSY\n")
                    conn.close()
                    continue

                self.client_socket = conn
                self.client_address = addr
                self.root.after(0, lambda a=addr: self._update_pi_status_ui(f"Raspberry Pi: {a[0]}:{a[1]} bağlandı."))
                self.root.after(0, lambda: self.btn_drive_vehicle.config(
                    state=tk.NORMAL if self.last_generated_commands_for_pi_json else tk.DISABLED))
                self.root.after(0, lambda: self.btn_stop_vehicle_on_pi.config(state=tk.NORMAL))

                self._handle_client_connection(conn, addr)

                if self.client_socket == conn:
                    self.client_socket = None
                    self.client_address = None
                    self.root.after(0, lambda: self._update_pi_status_ui("Raspberry Pi: Bağlantı Kesildi."))
                    self.root.after(0, lambda: self.btn_drive_vehicle.config(state=tk.DISABLED))
                    self.root.after(0, lambda: self.btn_stop_vehicle_on_pi.config(state=tk.DISABLED))
                    self.is_pi_calibrating = False
                    self.is_pi_driving = False

            except socket.timeout:
                continue
            except socket.error as e:
                if self.stop_socket_server_event.is_set():
                    print(f"Sunucu soketi kapatıldı (beklenen): {e}")
                else:
                    print(f"Sunucu accept/socket hatası: {e}")
                break 
            except Exception as e:
                if not self.stop_socket_server_event.is_set():
                     self.root.after(0, lambda msg=f"Sunucu döngü hatası: {e}": self._update_server_status_ui(msg))
                break

        if self.server_socket: 
            try:
                self.server_socket.close()
            except: pass
        self.server_socket = None 
        self.is_server_running = False 
        self.root.after(0, lambda: self.btn_toggle_server.config(text="Sunucuyu Başlat"))
        if not self.stop_socket_server_event.is_set(): 
            self.root.after(0, lambda: self._update_server_status_ui("Sunucu: Hata ile durdu"))
        else: 
             self.root.after(0, lambda: self._update_server_status_ui("Sunucu: Durduruldu"))
        print("Soket sunucu döngüsü sonlandı.")


    def _handle_client_connection(self, client_sock, client_addr):
        client_sock.settimeout(1.0)
        self.is_pi_calibrating = False
        self.is_pi_driving = False

        try:
            while self.is_server_running and not self.stop_socket_server_event.is_set():
                try:
                    data = client_sock.recv(1024)
                    if not data:
                        self.root.after(0,
                                        lambda: self._update_pi_status_ui(f"Pi ({client_addr[0]}) bağlantıyı kapattı."))
                        break

                    message_from_pi = data.decode('utf-8').strip()
                    for single_msg in message_from_pi.split('\n'):
                        if single_msg:
                            self.root.after(0, lambda msg=single_msg.strip(): self.process_message_from_pi(msg))

                except socket.timeout:
                    continue
                except socket.error as e:
                    self.root.after(0, lambda err=str(e): self._update_pi_status_ui(
                        f"Pi ({client_addr[0]}) ile soket hatası: {err}"))
                    break
        finally:
            print(f"İstemci {client_addr} ile bağlantı sonlandırılıyor (_handle_client_connection).")

    def send_to_pi(self, message):
        if self.client_socket and self.is_server_running:
            try:
                self.client_socket.sendall((message + "\n").encode('utf-8'))
                print(f"Pi'ye gönderildi: {message}")
                return True
            except socket.error as e:
                self._update_pi_status_ui(f"Pi'ye gönderme hatası: {e}")
                if self.client_socket:
                    try: self.client_socket.close()
                    except: pass
                self.client_socket = None
                self.client_address = None
                self._update_pi_status_ui("Raspberry Pi: Bağlantı Kesildi.")
                self.btn_drive_vehicle.config(state=tk.DISABLED)
                self.btn_stop_vehicle_on_pi.config(state=tk.DISABLED)
                self.is_pi_calibrating = False
                self.is_pi_driving = False
                return False
        else:
            self._update_pi_status_ui("Pi'ye gönderilemedi: Bağlantı yok veya sunucu çalışmıyor.")
            return False

    def process_message_from_pi(self, message):
        self._update_pi_status_ui(f"Pi'den alındı: {message}")

        if message.startswith("CALIBRATION_DONE:"):
            self.is_pi_calibrating = False
            try:
                offset_str = message.split(":")[1]
                self.pi_calibration_offset = float(offset_str)
                self._update_pi_status_ui(
                    f"Pi kalibrasyonu tamamlandı (Ofset: {self.pi_calibration_offset:.2f}). Komutlar gönderiliyor...")

                if not self.last_generated_commands_for_pi_json:
                    self._update_pi_status_ui("Hata: Pi için gönderilecek komut bulunamadı.")
                    self.send_to_pi("ERROR:NO_COMMANDS_AVAILABLE")
                    return

                if self.last_generated_commands_for_pi_json:
                    json_commands = json.dumps(self.last_generated_commands_for_pi_json)
                    self.send_to_pi(f"COMMANDS:{json_commands}")
                    self.is_pi_driving = True
                    self.btn_drive_vehicle.config(state=tk.DISABLED)
                    self.btn_stop_vehicle_on_pi.config(state=tk.NORMAL)
                else: 
                    self._update_pi_status_ui("Pi için geçerli komut oluşturulamadı (iç boş).")
                    self.send_to_pi("ERROR:NO_VALID_COMMANDS_FORMED_EMPTY")

            except Exception as e:
                self._update_pi_status_ui(f"Kalibrasyon ofset ayrıştırma veya komut gönderme hatası: {e}")
                self.send_to_pi("ERROR:CALIBRATION_OR_COMMAND_SEND_ERROR")


        elif message == "CALIBRATION_FAIL:MPU_INIT_ERROR":
            self.is_pi_calibrating = False
            self._update_pi_status_ui("Pi kalibrasyonu başarısız: MPU başlatılamadı.")
            self.btn_drive_vehicle.config(
                state=tk.NORMAL if self.client_socket and self.last_generated_commands_for_pi_json else tk.DISABLED)

        elif message == "COMMANDS_RECEIVED_VALID":
            self._update_pi_status_ui("Pi komutları aldı, araç hareket ediyor...")
            self.btn_stop_vehicle_on_pi.config(state=tk.NORMAL)

        elif message == "COMMANDS_INVALID_FORMAT":
            self.is_pi_driving = False
            self._update_pi_status_ui("Pi komut formatını geçersiz buldu.")
            self.btn_drive_vehicle.config(
                state=tk.NORMAL if self.client_socket and self.last_generated_commands_for_pi_json else tk.DISABLED)
            self.btn_stop_vehicle_on_pi.config(state=tk.DISABLED)

        elif message == "SEQUENCE_DONE":
            self.is_pi_driving = False
            self._update_pi_status_ui("Pi komut dizisini tamamladı.")
            self.btn_drive_vehicle.config(
                state=tk.NORMAL if self.client_socket and self.last_generated_commands_for_pi_json else tk.DISABLED)
            self.btn_stop_vehicle_on_pi.config(state=tk.DISABLED)

        elif message == "STOP_ACK":
            self.is_pi_driving = False
            self.is_pi_calibrating = False
            self._update_pi_status_ui("Pi aracı durdurdu (STOP_ACK).")
            self.btn_drive_vehicle.config(
                state=tk.NORMAL if self.client_socket and self.last_generated_commands_for_pi_json else tk.DISABLED)
            self.btn_stop_vehicle_on_pi.config(state=tk.DISABLED)


    def drive_vehicle_on_pi(self):
        if not self.client_socket:
            messagebox.showerror("Bağlantı Hatası", "Raspberry Pi bağlı değil.")
            return
        if not self.last_generated_commands_for_pi_json: # Pi için JSON formatındaki komutları kontrol et
            messagebox.showerror("Komut Hatası", "Önce 'Yolu Bul ve İşle' ile komutları oluşturun.")
            return
        if self.is_pi_calibrating or self.is_pi_driving:
            messagebox.showwarning("İşlem Sürüyor", "Pi zaten kalibrasyon yapıyor veya aracı sürüyor.")
            return
        if not self.current_image_source:
            messagebox.showerror("Hata", "Görüntü kaynağı belirlenemedi (galeri/kamera). Lütfen bir görüntü yükleyin.")
            return

        self.is_pi_calibrating = True
        calibrate_command = f"CALIBRATE:{self.current_image_source.upper()}"
        self.send_to_pi(calibrate_command)
        self._update_pi_status_ui(f"Pi'ye kalibrasyon komutu gönderildi ({self.current_image_source}). Bekleniyor...")
        self.btn_drive_vehicle.config(state=tk.DISABLED)
        self.btn_stop_vehicle_on_pi.config(state=tk.NORMAL)

    def stop_vehicle_on_pi(self):
        if not self.client_socket:
            messagebox.showerror("Bağlantı Hatası", "Raspberry Pi bağlı değil.")
            return
        self.send_to_pi("STOP")
        self._update_pi_status_ui("Pi'ye DUR komutu gönderildi.")
        self.btn_drive_vehicle.config(state=tk.DISABLED)
        self.btn_stop_vehicle_on_pi.config(state=tk.DISABLED)

    def start_progress(self):
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.start(10)

    def stop_progress(self):
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.stop()
            self.progress_bar['value'] = 0

    def on_results_tab_changed(self, event):
        try:
            if not self.notebook_results.winfo_exists(): return
            selected_tab_id = self.notebook_results.select()
            if not selected_tab_id: return
            selected_tab_text = self.notebook_results.tab(selected_tab_id, "text")

            if selected_tab_text == 'Canlı Kamera & Yol':
                self.stop_path_animation()
                self.stop_command_animation()
                if self.last_simplified_path_for_overlay or self.last_generated_commands_for_display:
                    self.start_live_camera_feed()
                else:
                    if hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
                        self.live_camera_status_label.config(text="Canlı yol için önce bir yol bulunmalı.")
            elif selected_tab_text == 'Çözülmüş Labirent':
                self.stop_live_camera_feed()
            else: 
                self.stop_live_camera_feed()
                self.stop_path_animation()
                self.stop_command_animation()
        except tk.TclError as e:
            print(f"Sekme değiştirme sırasında TclError: {e} (muhtemelen widget yok edildi).")
        except Exception as e:
            print(f"on_results_tab_changed içinde bir HATA oluştu: {e}")
            traceback.print_exc()

    def on_generic_canvas_resize_wrapper(self, event, canvas_widget, static_pil_attr_name, dynamic_pil_attr_name):
        if (self.is_path_animating or self.is_command_animating) and \
                hasattr(self, dynamic_pil_attr_name) and getattr(self, dynamic_pil_attr_name):
            self.on_generic_canvas_resize(event, canvas_widget, dynamic_pil_attr_name)
        elif hasattr(self, static_pil_attr_name) and getattr(self, static_pil_attr_name):
            self.on_generic_canvas_resize(event, canvas_widget, static_pil_attr_name)
        elif hasattr(self, static_pil_attr_name) and getattr(self, static_pil_attr_name) is None and canvas_widget.winfo_exists():
            canvas_widget.delete("all")
            if hasattr(canvas_widget, "_current_tk_image_ref"):
                try: delattr(canvas_widget, "_current_tk_image_ref")
                except (AttributeError, tk.TclError): pass


    def on_closing(self):
        print("Uygulama kapatılıyor...")
        self.stop_path_animation()
        self.stop_command_animation()
        self.stop_live_camera_feed()

        if self.is_camera_streaming_on_main_canvas:
            self.stop_camera_preview_event.set()
            if self.camera_preview_thread and self.camera_preview_thread.is_alive():
                self.camera_preview_thread.join(timeout=1.0)
        if self.video_capture_device_main_canvas and self.video_capture_device_main_canvas.isOpened():
            self.video_capture_device_main_canvas.release()
        self.video_capture_device_main_canvas = None

        if self.video_capture_device and self.video_capture_device.isOpened(): 
            self.video_capture_device.release()
        self.video_capture_device = None

        if self.is_server_running:
            self.stop_socket_server_event.set()
            if self.client_socket:
                try: self.client_socket.sendall(b"SERVER_SHUTDOWN\n")
                except: pass
                try: self.client_socket.close()
                except: pass
            if self.server_socket:
                try: self.server_socket.close()
                except: pass
            if self.socket_server_thread and self.socket_server_thread.is_alive():
                self.socket_server_thread.join(timeout=1.0)

        self.client_socket = None
        self.server_socket = None

        if self.root.winfo_exists():
            self.root.destroy()

    def reset_all_app_state(self):
        print("reset_all_app_state çağrıldı.")
        self.stop_path_animation(); self.stop_command_animation(); self.stop_live_camera_feed()

        if self.is_camera_streaming_on_main_canvas:
            self.stop_camera_preview_event.set()
            if self.camera_preview_thread and self.camera_preview_thread.is_alive():
                try: self.camera_preview_thread.join(timeout=0.2)
                except Exception as e: print(f"reset_all_app_state'de kamera thread join hatası: {e}")
        self.is_camera_streaming_on_main_canvas = False
        self.current_preview_frame_cv2 = None
        self.stop_camera_preview_event.clear()
        self.camera_preview_thread = None
        if self.video_capture_device_main_canvas and self.video_capture_device_main_canvas.isOpened():
            self.video_capture_device_main_canvas.release()
        self.video_capture_device_main_canvas = None

        if self.is_server_running:
            if self.client_socket:
                self.stop_vehicle_on_pi()
                time.sleep(0.5)
            self.toggle_socket_server()

        self._update_pi_status_ui("Raspberry Pi: Bağlı Değil")
        self.is_pi_calibrating = False; self.is_pi_driving = False; self.pi_calibration_offset = None

        self.image_path = None; self.original_cv_image = None; self.h_orig_for_path, self.w_orig_for_path = 0, 0
        self.image_with_skeleton_overlay_for_selection = None; self.display_image_tk = None; self.displayed_image_pil = None
        self.current_image_source = None
        self.mask_for_bfs_and_clicking_ORIG_SCALE = None; self.padding_info = {}
        self.start_point_original_coords = None; self.end_point_original_coords = None
        self.selected_points_on_canvas.clear()

        self.current_mask_type_str = "Bilinmeyen"
        self.last_simplified_path_for_overlay = None
        self.last_generated_commands_for_pi_json = []
        self.last_generated_commands_for_display = []


        self.result_image_display_tk_pil = None; self.bfs_mask_pil = None
        self.raw_model_mask_pil = None; self.skeleton_model_scale_pil = None
        self.current_animation_frame_pil = None

        for canvas_attr_name in ['canvas_image', 'lbl_result_image_canvas', 'lbl_skeleton_mask_canvas',
                                 'lbl_raw_model_mask_canvas', 'live_camera_canvas']:
            canvas = getattr(self, canvas_attr_name, None)
            if canvas and canvas.winfo_exists():
                if hasattr(canvas, "_current_tk_image_ref"):
                    try: delattr(canvas, "_current_tk_image_ref")
                    except: pass
                if hasattr(canvas, "_current_tk_image_ref_preview"): 
                    try: delattr(canvas, "_current_tk_image_ref_preview")
                    except: pass
                try: canvas.delete("all")
                except tk.TclError: pass

        if hasattr(self, 'txt_commands') and self.txt_commands.winfo_exists(): self.txt_commands.delete(1.0, tk.END)
        if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists(): self.lbl_image_status.config(text="Görüntü seçilmedi.")
        if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists(): self.lbl_point_instruction.config(text="1. Başlangıç noktasını seçin.")

        if hasattr(self, 'btn_load_camera') and self.btn_load_camera.winfo_exists(): self.btn_load_camera.config(text="Kameradan Görüntü Al", state=tk.NORMAL)
        if hasattr(self, 'btn_load_gallery') and self.btn_load_gallery.winfo_exists(): self.btn_load_gallery.config(state=tk.NORMAL)
        if hasattr(self, 'btn_process') and self.btn_process.winfo_exists(): self.btn_process.config(state=tk.DISABLED)
        if hasattr(self, 'btn_animate_path') and self.btn_animate_path.winfo_exists(): self.btn_animate_path.config(state=tk.DISABLED)
        if hasattr(self, 'btn_animate_commands') and self.btn_animate_commands.winfo_exists(): self.btn_animate_commands.config(state=tk.DISABLED)
        if hasattr(self, 'btn_drive_vehicle') and self.btn_drive_vehicle.winfo_exists(): self.btn_drive_vehicle.config(state=tk.DISABLED)
        if hasattr(self, 'btn_stop_vehicle_on_pi') and self.btn_stop_vehicle_on_pi.winfo_exists(): self.btn_stop_vehicle_on_pi.config(state=tk.DISABLED)
        if hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists(): self.live_camera_status_label.config(text="Yol bulunursa kamera burada aktifleşir.")
        print("Sistem sıfırlandı.")


    def load_model_on_startup(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Model Hatası", f"Model dosyası bulunamadı: {MODEL_PATH}")
            self.root.quit()
            return
        try:
            print("Model yükleniyor...")
            self.start_progress()
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.input_shape_tuple = self.model.input_shape[1:]
            self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS = self.input_shape_tuple
            self.is_grayscale_model = (self.IMG_CHANNELS == 1)
            print(
                f"'{MODEL_PATH}' modeli yüklendi. Girdi: {self.IMG_HEIGHT}x{self.IMG_WIDTH}x{self.IMG_CHANNELS} (Gri: {self.is_grayscale_model})")
        except Exception as e:
            messagebox.showerror("Model Yükleme Hatası",
                                 f"Model yüklenirken bir hata oluştu:\n{e}\n{traceback.format_exc()}")
            self.root.quit()
        finally:
            self.stop_progress()

    def load_vehicle_image(self):
        try:
            if os.path.exists(VEHICLE_IMAGE_FILENAME):
                pil_image = Image.open(VEHICLE_IMAGE_FILENAME).convert("RGBA")
                target_height = 35
                aspect_ratio = pil_image.width / pil_image.height
                target_width = int(target_height * aspect_ratio)
                self.original_vehicle_pil = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            else:
                print(f"UYARI: Araç görseli '{VEHICLE_IMAGE_FILENAME}' bulunamadı.")
                self.original_vehicle_pil = None
        except Exception as e:
            print(f"Araç görseli yüklenirken hata: {e}")
            self.original_vehicle_pil = None

    def on_main_canvas_resize(self, event=None):
        if not self.is_camera_streaming_on_main_canvas and self.image_with_skeleton_overlay_for_selection is not None:
            self._update_main_canvas_display(self.image_with_skeleton_overlay_for_selection)

    def on_generic_canvas_resize(self, event, canvas_widget, pil_image_attr_name):
        pil_image_original = getattr(self, pil_image_attr_name, None)
        if pil_image_original and canvas_widget.winfo_exists():
            canvas_width = canvas_widget.winfo_width()
            canvas_height = canvas_widget.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1: return 

            orig_w, orig_h = pil_image_original.size
            if orig_w == 0 or orig_h == 0: return

            padding = 10 
            target_box_w = canvas_width - padding
            target_box_h = canvas_height - padding
            if target_box_w <=0 : target_box_w = 1
            if target_box_h <=0 : target_box_h = 1


            scale = min(target_box_w / orig_w, target_box_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            if new_w < 1: new_w = 1
            if new_h < 1: new_h = 1

            try:
                img_copy = pil_image_original.copy()
                resampling_method = Image.Resampling.NEAREST if "mask" in pil_image_attr_name.lower() else Image.Resampling.LANCZOS
                if scale > 1.5 and "mask" in pil_image_attr_name.lower() : resampling_method = Image.Resampling.NEAREST
                elif scale < 0.5 : resampling_method = Image.Resampling.LANCZOS 

                resized_image = img_copy.resize((new_w, new_h), resampling_method)
                tk_image = ImageTk.PhotoImage(image=resized_image)

                setattr(canvas_widget, "_current_tk_image_ref", tk_image) 
                canvas_widget.delete("all")
                canvas_widget.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=tk_image)
            except Exception as e:
                print(f"on_generic_canvas_resize HATA ({pil_image_attr_name}): {e}")
                traceback.print_exc()
        elif canvas_widget.winfo_exists(): 
            canvas_widget.delete("all")
            if hasattr(canvas_widget, "_current_tk_image_ref"):
                try: delattr(canvas_widget, "_current_tk_image_ref")
                except (AttributeError, tk.TclError): pass


    def _update_main_canvas_display(self, cv_image_to_show):
        if cv_image_to_show is None: return
        try:
            img_rgb = cv2.cvtColor(cv_image_to_show,
                                   cv2.COLOR_BGR2RGB if len(cv_image_to_show.shape) == 3 else cv2.COLOR_GRAY2RGB)
            pil_image_orig_for_main_canvas = Image.fromarray(img_rgb)

            if not self.canvas_image.winfo_exists(): return

            canvas_width = self.canvas_image.winfo_width()
            canvas_height = self.canvas_image.winfo_height()

            if canvas_width > 1 and canvas_height > 1: 
                img_copy = pil_image_orig_for_main_canvas.copy()
                img_copy.thumbnail((canvas_width - 10, canvas_height - 10), Image.Resampling.LANCZOS) 
                self.displayed_image_pil = img_copy
            else: 
                self.displayed_image_pil = pil_image_orig_for_main_canvas

            self.display_image_tk = ImageTk.PhotoImage(image=self.displayed_image_pil)
            self.canvas_image.delete("all")

            cw_final = self.canvas_image.winfo_width()
            ch_final = self.canvas_image.winfo_height()
            if cw_final > 0 and ch_final > 0 :
                self.canvas_image.create_image(cw_final // 2, ch_final // 2, anchor=tk.CENTER, image=self.display_image_tk)
            else: 
                self.canvas_image.create_image(0, 0, anchor=tk.NW, image=self.display_image_tk)


            self.redraw_selected_points() 
        except Exception as e:
            print(f"_update_main_canvas_display HATA: {e}")
            traceback.print_exc()

    def redraw_selected_points(self):
        if not self.canvas_image.winfo_exists(): return

        for p_id in self.selected_points_on_canvas:
            self.canvas_image.delete(p_id)
        self.selected_points_on_canvas.clear()

        if not self.displayed_image_pil or self.h_orig_for_path == 0 or self.w_orig_for_path == 0:
            return # Gerekli bilgi yoksa çizme
        if self.displayed_image_pil.width == 0 or self.displayed_image_pil.height == 0:
            return


        scale_x = self.displayed_image_pil.width / self.w_orig_for_path
        scale_y = self.displayed_image_pil.height / self.h_orig_for_path

        canvas_width = self.canvas_image.winfo_width()
        canvas_height = self.canvas_image.winfo_height()
        offset_x = (canvas_width - self.displayed_image_pil.width) / 2
        offset_y = (canvas_height - self.displayed_image_pil.height) / 2


        points_to_draw = []
        if self.start_point_original_coords:
            points_to_draw.append((self.start_point_original_coords, "green"))
        if self.end_point_original_coords:
            points_to_draw.append((self.end_point_original_coords, "red"))

        for (orig_coords, color) in points_to_draw:
            orig_y, orig_x = orig_coords 
            canvas_x = int(orig_x * scale_x + offset_x)
            canvas_y = int(orig_y * scale_y + offset_y)
            p_id = self.canvas_image.create_oval(canvas_x - 5, canvas_y - 5, canvas_x + 5, canvas_y + 5, fill=color,
                                           outline="black", width=2)
            self.selected_points_on_canvas.append(p_id)


    def _load_and_preprocess_image_for_model_and_mask(self):
        if not self.image_path or not os.path.exists(self.image_path):
            messagebox.showerror("Hata", "Geçerli bir görüntü yolu belirtilmemiş.")
            return False
        try:
            self.start_progress()
            self.original_cv_image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if self.original_cv_image is None:
                messagebox.showerror("Hata", f"Görüntü yüklenemedi: {os.path.basename(self.image_path)}")
                return False
            self.h_orig_for_path, self.w_orig_for_path = self.original_cv_image.shape[:2]

            if self.model is None or self.IMG_HEIGHT is None:
                messagebox.showerror("Hata", "Model yüklenmemiş.")
                return False

            read_mode = cv2.IMREAD_GRAYSCALE if self.is_grayscale_model else cv2.IMREAD_COLOR
            img_to_process_for_model = cv2.imread(self.image_path, read_mode)
            if img_to_process_for_model is None:
                messagebox.showerror("Hata", f"Girdi görüntüsü okunamadı: {os.path.basename(self.image_path)}")
                return False

            target_h, target_w = self.IMG_HEIGHT, self.IMG_WIDTH
            current_h_m, current_w_m = img_to_process_for_model.shape[:2]

            scale = 1.0
            if current_w_m > 0 and current_h_m > 0: 
                scale = min(target_w / current_w_m, target_h / current_h_m)

            new_w, new_h = int(current_w_m * scale), int(current_h_m * scale)
            if new_w == 0 or new_h == 0: 
                 messagebox.showerror("Hata", "Görüntü ölçekleme hatası (boyut sıfır).")
                 return False


            resized_img = cv2.resize(img_to_process_for_model, (new_w, new_h), interpolation=cv2.INTER_AREA)

            top_pad = (target_h - new_h) // 2
            bottom_pad = target_h - new_h - top_pad
            left_pad = (target_w - new_w) // 2
            right_pad = target_w - new_w - left_pad
            self.padding_info = {
                'top_pad': top_pad, 'left_pad': left_pad,
                'new_w': new_w, 'new_h': new_h, 
                'bottom_pad': bottom_pad, 'right_pad': right_pad,
                'original_model_input_shape': (target_h, target_w) 
            }
            border_val = 0 if self.is_grayscale_model else [0,0,0]
            padded_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=border_val)

            img_processed_for_model = np.expand_dims(padded_img, axis=-1) if self.is_grayscale_model and padded_img.ndim == 2 else padded_img

            if img_processed_for_model.shape[0] != target_h or img_processed_for_model.shape[1] != target_w:
                print(f"UYARI: Dolgulu görüntü boyutu ({img_processed_for_model.shape}) model girdisiyle ({target_h},{target_w}) eşleşmiyor. Yeniden boyutlandırılıyor.")
                img_processed_for_model = cv2.resize(img_processed_for_model, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                if self.is_grayscale_model and img_processed_for_model.ndim == 2:
                    img_processed_for_model = np.expand_dims(img_processed_for_model, axis=-1)


            prediction = self.model.predict(np.expand_dims(img_processed_for_model / 255.0, axis=0))
            predicted_mask_prob = prediction[0] 
            if predicted_mask_prob.shape[-1] > 1: 
                predicted_mask_prob = predicted_mask_prob[:,:,0:1]


            mask_3d_model_scale = (predicted_mask_prob < THRESHOLD).astype(np.uint8) 
            mask_from_model_prediction_MODEL_SCALE = np.squeeze(mask_3d_model_scale).astype(np.uint8) 
            self.raw_model_mask_pil = Image.fromarray((mask_from_model_prediction_MODEL_SCALE * 255).astype(np.uint8))


            self.skeleton_model_scale_pil = None
            chosen_mask_for_processing_MODEL_SCALE = mask_from_model_prediction_MODEL_SCALE.copy()
            self.current_mask_type_str = "Orijinal Model Maskesi (BFS için)"

            if SKIMAGE_AVAILABLE:
                try:
                    skeleton_model_scale_01 = skeletonize(chosen_mask_for_processing_MODEL_SCALE == 1).astype(np.uint8) 
                    if np.any(skeleton_model_scale_01): 
                        self.skeleton_model_scale_pil = Image.fromarray((skeleton_model_scale_01 * 255).astype(np.uint8))
                        chosen_mask_for_processing_MODEL_SCALE = skeleton_model_scale_01
                        self.current_mask_type_str = "İskelet Maskesi (BFS için)"
                    else:
                        self.current_mask_type_str += " (İskelet Boş, BFS orijinali kullanıyor)"
                except Exception as e_skele:
                    print(f"İskelet çıkarma hatası: {e_skele}")
                    self.current_mask_type_str += f" (İskelet Hatası, BFS orijinali kullanıyor)"


            pi = self.padding_info
            cropped_mask = chosen_mask_for_processing_MODEL_SCALE[
                           pi['top_pad']:pi['original_model_input_shape'][0] - pi['bottom_pad'],
                           pi['left_pad']:pi['original_model_input_shape'][1] - pi['right_pad']]

            if cropped_mask.shape[0] != pi['new_h'] or cropped_mask.shape[1] != pi['new_w']:
                if pi['new_h'] > 0 and pi['new_w'] > 0:
                     cropped_mask = cv2.resize(cropped_mask, (pi['new_w'], pi['new_h']), interpolation=cv2.INTER_NEAREST)
                else: 
                    print(f"UYARI: Cropped_mask için geçersiz new_h/new_w: {pi['new_h']}x{pi['new_w']}. Tam maske yeniden boyutlandırılıyor.")
                    if self.w_orig_for_path > 0 and self.h_orig_for_path > 0:
                        self.mask_for_bfs_and_clicking_ORIG_SCALE = cv2.resize(chosen_mask_for_processing_MODEL_SCALE, (self.w_orig_for_path, self.h_orig_for_path), interpolation=cv2.INTER_NEAREST)
                    else:
                        messagebox.showerror("Hata", "Maske boyutlandırmada kritik hata: Orijinal veya yeni boyutlar sıfır.")
                        return False


            if pi['new_h'] > 0 and pi['new_w'] > 0: 
                if cropped_mask.size > 0: 
                    self.mask_for_bfs_and_clicking_ORIG_SCALE = cv2.resize(cropped_mask,
                                                                       (self.w_orig_for_path, self.h_orig_for_path),
                                                                       interpolation=cv2.INTER_NEAREST)
                else:
                     self.mask_for_bfs_and_clicking_ORIG_SCALE = cv2.resize(chosen_mask_for_processing_MODEL_SCALE,
                                                                       (self.w_orig_for_path, self.h_orig_for_path),
                                                                       interpolation=cv2.INTER_NEAREST)


            if self.mask_for_bfs_and_clicking_ORIG_SCALE is None:
                messagebox.showerror("Hata", "BFS için maske oluşturulamadı (son aşama).")
                return False

            self.bfs_mask_pil = Image.fromarray((self.mask_for_bfs_and_clicking_ORIG_SCALE * 255).astype(np.uint8))


            self.image_with_skeleton_overlay_for_selection = self.original_cv_image.copy()
            overlay_color = [0, 255, 255]  
            path_pixels = self.mask_for_bfs_and_clicking_ORIG_SCALE == 1 

            highlight_overlay_temp = np.zeros_like(self.image_with_skeleton_overlay_for_selection)
            highlight_overlay_temp[path_pixels] = overlay_color
            self.image_with_skeleton_overlay_for_selection = cv2.addWeighted(self.image_with_skeleton_overlay_for_selection, 0.7, highlight_overlay_temp, 0.3, 0)


            return True
        except Exception as e:
            messagebox.showerror("Görüntü Ön İşleme Hatası",
                                 f"Görüntü işlenirken bir hata oluştu:\n{e}\n{traceback.format_exc()}")
            return False
        finally:
            self.stop_progress()

    def load_image_from_gallery(self):
        self.stop_path_animation(); self.stop_command_animation(); self.stop_live_camera_feed()
        if self.is_camera_streaming_on_main_canvas:
            self.capture_frame_from_preview_and_process() 

        new_path = filedialog.askopenfilename(
            title="Labirent Görüntüsü Seçin",
            filetypes=(("Resim Dosyaları", "*.jpg *.jpeg *.png"), ("Tüm Dosyalar", "*.*"))
        )
        if new_path:
            self.image_path = new_path
            self.current_image_source = "gallery" 
            self._reset_image_specific_state() 
            self._proceed_with_image_loading()


    def _reset_image_specific_state(self):
        """Sadece görüntü ve yol bulmayla ilgili durumları sıfırlar."""
        self.stop_path_animation()
        self.stop_command_animation()

        self.original_cv_image = None
        self.h_orig_for_path, self.w_orig_for_path = 0, 0
        self.image_with_skeleton_overlay_for_selection = None
        self.display_image_tk = None
        self.displayed_image_pil = None

      self.mask_for_bfs_and_clicking_ORIG_SCALE = None
        self.padding_info = {}
        self.start_point_original_coords = None
        self.end_point_original_coords = None
        self.selected_points_on_canvas.clear()

        self.current_mask_type_str = "Bilinmeyen"
        self.last_simplified_path_for_overlay = None
        self.last_generated_commands_for_pi_json = [] 
        self.last_generated_commands_for_display = [] 


        self.result_image_display_tk_pil = None
        self.bfs_mask_pil = None
        self.raw_model_mask_pil = None
        self.skeleton_model_scale_pil = None
        self.current_animation_frame_pil = None

        for canvas_attr_name in ['canvas_image', 'lbl_result_image_canvas', 'lbl_skeleton_mask_canvas',
                                 'lbl_raw_model_mask_canvas']: 
            canvas = getattr(self, canvas_attr_name, None)
            if canvas and canvas.winfo_exists():
                if hasattr(canvas, "_current_tk_image_ref"):
                    try: delattr(canvas, "_current_tk_image_ref")
                    except: pass
                try: canvas.delete("all")
                except tk.TclError: pass

        if hasattr(self, 'txt_commands') and self.txt_commands.winfo_exists(): self.txt_commands.delete(1.0, tk.END)

        if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists(): self.lbl_image_status.config(text="Görüntü seçilmedi.")
        if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists(): self.lbl_point_instruction.config(text="1. Başlangıç noktasını seçin.")

        if hasattr(self, 'btn_process') and self.btn_process.winfo_exists(): self.btn_process.config(state=tk.DISABLED)
        if hasattr(self, 'btn_animate_path') and self.btn_animate_path.winfo_exists(): self.btn_animate_path.config(state=tk.DISABLED)
        if hasattr(self, 'btn_animate_commands') and self.btn_animate_commands.winfo_exists(): self.btn_animate_commands.config(state=tk.DISABLED)
        if hasattr(self, 'btn_drive_vehicle') and self.btn_drive_vehicle.winfo_exists():
            self.btn_drive_vehicle.config(state=tk.DISABLED)


    def toggle_camera_stream_and_capture(self):
        if self.is_camera_streaming_on_main_canvas:
            print("Kamera butonuyla yakalama tetiklendi.")
            self.capture_frame_from_preview_and_process()
        else:
            print("Kamera butonuyla önizleme başlatılıyor.")
            self._reset_image_specific_state()
            self.current_image_source = "camera" 

            self.is_camera_streaming_on_main_canvas = True
            self.stop_camera_preview_event.clear()
            self.current_preview_frame_cv2 = None

            if hasattr(self, 'canvas_image') and self.canvas_image.winfo_exists():
                self.canvas_image.delete("all") 
            self.selected_points_on_canvas.clear()
            self.start_point_original_coords = None
            self.end_point_original_coords = None


            if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists():
                self.lbl_point_instruction.config(text="Kamera aktif. Yakalamak için tıklayın veya butona basın.")
            if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists():
                self.lbl_image_status.config(text="Kamera önizlemesi...")


            self.btn_load_camera.config(text="Görüntü Yakala")
            self.btn_load_gallery.config(state=tk.DISABLED) 
            self.btn_process.config(state=tk.DISABLED) 

            self.camera_preview_thread = threading.Thread(target=self._camera_preview_loop, daemon=True)
            self.camera_preview_thread.start()

    def capture_frame_from_preview_and_process(self):
        if not self.is_camera_streaming_on_main_canvas :
            print("Yakalamaya çalışıldı ancak kamera önizlemesi aktif değil veya zaten yakalandı.")
            self._handle_preview_error_ui_reset() 
            return

        print("Görüntü yakalanıyor ve işleniyor...")
        self.is_camera_streaming_on_main_canvas = False 
        self.stop_camera_preview_event.set()

        captured_frame_to_process = self.current_preview_frame_cv2 

        if self.camera_preview_thread and self.camera_preview_thread.is_alive():
            print("Ana kanvas kamera thread'inin sonlanması bekleniyor (capture)...")
            self.camera_preview_thread.join(timeout=1.0) 
        self.camera_preview_thread = None 

        if hasattr(self, 'btn_load_camera') and self.btn_load_camera.winfo_exists():
            self.btn_load_camera.config(text="Kameradan Görüntü Al", state=tk.NORMAL)
        if hasattr(self, 'btn_load_gallery') and self.btn_load_gallery.winfo_exists():
            self.btn_load_gallery.config(state=tk.NORMAL)


        if captured_frame_to_process is not None:
            temp_image_filename = "temp_camera_capture.jpg"
            save_success = cv2.imwrite(temp_image_filename, captured_frame_to_process)
            if save_success:
                self.image_path = temp_image_filename
                self.current_image_source = "camera" 
                self.current_preview_frame_cv2 = None 
                self._reset_image_specific_state() 
                self._proceed_with_image_loading()
            else:
                messagebox.showerror("Kamera Hatası", "Yakalanan görüntü diske yazılamadı.")
                if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists(): self.lbl_image_status.config(text="Görüntü seçilmedi.")
                if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists(): self.lbl_point_instruction.config(text="1. Başlangıç noktasını seçin.")
                self.current_preview_frame_cv2 = None
                self.current_image_source = None
        else:
            messagebox.showerror("Kamera Hatası", "Kameradan görüntü yakalanamadı (kare boş).")
            if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists(): self.lbl_image_status.config(text="Görüntü seçilmedi.")
            if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists(): self.lbl_point_instruction.config(text="1. Başlangıç noktasını seçin.")
            self.current_preview_frame_cv2 = None
            self.current_image_source = None


    def _camera_preview_loop(self):
        try:
            if self.video_capture_device_main_canvas is None or not self.video_capture_device_main_canvas.isOpened():
                print("Ana kanvas için kamera açılıyor...")
                self.video_capture_device_main_canvas = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
                if not (self.video_capture_device_main_canvas and self.video_capture_device_main_canvas.isOpened()):
                     self.video_capture_device_main_canvas = cv2.VideoCapture(1, cv2.CAP_DSHOW) 

            if not (self.video_capture_device_main_canvas and self.video_capture_device_main_canvas.isOpened()):
                if self.root.winfo_exists():
                    self.root.after(0, lambda: messagebox.showerror("Kamera Hatası", "Ana önizleme için kamera açılamadı."))
                    self.root.after(0, self._handle_preview_error_ui_reset)
                self.is_camera_streaming_on_main_canvas = False 
                return
            print("Ana kanvas kamera önizlemesi başladı.")
            print("Ana kanvas kamera önizlemesi başladı.")
            while not self.stop_camera_preview_event.is_set():
                if not (self.video_capture_device_main_canvas and self.video_capture_device_main_canvas.isOpened()):
                    print("Ana kanvas kamera önizleme döngüsü: Cihaz kapalı veya hata.")
                    break
                ret, frame = self.video_capture_device_main_canvas.read()
                if not ret:
                    print("Ana önizleme kamerasından kare okunamadı.")
                    self.stop_camera_preview_event.wait(0.01) 
                    continue

                self.current_preview_frame_cv2 = frame.copy() 

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image_orig = Image.fromarray(img_rgb)

                def update_main_preview_canvas_ui_thread(pil_img_to_show_preview):
                    if not self.root.winfo_exists() or not hasattr(self,'canvas_image') or \
                       not self.canvas_image.winfo_exists() or not self.is_camera_streaming_on_main_canvas: 
                        return 
                    try:
                        canvas_width = self.canvas_image.winfo_width()
                        canvas_height = self.canvas_image.winfo_height()

                        if canvas_width <= 1 or canvas_height <= 1: 
                             tk_image_preview = ImageTk.PhotoImage(image=pil_img_to_show_preview)
                        else:
                            img_copy_preview = pil_img_to_show_preview.copy()
                            img_copy_preview.thumbnail((canvas_width - 2, canvas_height - 2), Image.Resampling.LANCZOS) 
                            tk_image_preview = ImageTk.PhotoImage(image=img_copy_preview)

                        setattr(self.canvas_image, "_current_tk_image_ref_preview", tk_image_preview)
                        self.canvas_image.delete("all") 
                        self.canvas_image.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=tk_image_preview)
                    except Exception as e_ui_update:
                        if "application has been destroyed" in str(e_ui_update).lower() or \
                           "invalid command name" in str(e_ui_update).lower():
                            self.stop_camera_preview_event.set() 


                if self.root.winfo_exists(): 
                    self.root.after(0, lambda img_pil=pil_image_orig: update_main_preview_canvas_ui_thread(img_pil))
                else: 
                    break

                self.stop_camera_preview_event.wait(timeout=0.033) 
        except Exception as e_cam_loop:
            print(f"Ana kamera önizleme döngüsü HATA: {e_cam_loop}")
            traceback.print_exc()
            if self.root.winfo_exists(): 
                self.root.after(0, self._handle_preview_error_ui_reset)
        finally:
            print("Ana kamera önizleme döngüsü sonlanıyor.")
            if self.video_capture_device_main_canvas and self.video_capture_device_main_canvas.isOpened():
                print("Ana önizleme için video yakalama cihazı serbest bırakılıyor (_camera_preview_loop).")
                self.video_capture_device_main_canvas.release()

            if self.is_camera_streaming_on_main_canvas: 
                self.is_camera_streaming_on_main_canvas = False
                if self.root.winfo_exists():
                    self.root.after(0, self._handle_preview_error_ui_reset)


    def _handle_preview_error_ui_reset(self):
        print("Kamera önizleme hatası UI sıfırlaması.")
        if not self.root.winfo_exists(): return

        self.is_camera_streaming_on_main_canvas = False
        self.current_image_source = None
        if hasattr(self, 'btn_load_camera') and self.btn_load_camera.winfo_exists():
            self.btn_load_camera.config(text="Kameradan Görüntü Al", state=tk.NORMAL)
        if hasattr(self, 'btn_load_gallery') and self.btn_load_gallery.winfo_exists():
            self.btn_load_gallery.config(state=tk.NORMAL)
        if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists():
            self.lbl_image_status.config(text="Kamera hatası veya durduruldu.")
        if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists():
            self.lbl_point_instruction.config(text="1. Başlangıç noktasını seçin.")
        if hasattr(self, 'canvas_image') and self.canvas_image.winfo_exists():
            self.canvas_image.delete("all")
            if hasattr(self.canvas_image, "_current_tk_image_ref_preview"): 
                try: delattr(self.canvas_image, "_current_tk_image_ref_preview")
                except: pass
        if hasattr(self, 'btn_process') and self.btn_process.winfo_exists():
             self.btn_process.config(state=tk.DISABLED)


    def _proceed_with_image_loading(self):
        if not self.image_path: return

        if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists():
            self.lbl_image_status.config(text=f"Yükleniyor: {os.path.basename(self.image_path)} (Kaynak: {self.current_image_source})")
        self.root.update_idletasks() 

        load_success = self._load_and_preprocess_image_for_model_and_mask()

        if load_success:
            if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists():
                self.lbl_image_status.config(text=f"Hazır: {os.path.basename(self.image_path)} (Kaynak: {self.current_image_source})")
            self._update_main_canvas_display(self.image_with_skeleton_overlay_for_selection)
            if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists():
                self.lbl_point_instruction.config(text="1. Başlangıç noktasını seçin.")
            if hasattr(self, 'btn_process') and self.btn_process.winfo_exists():
                self.btn_process.config(state=tk.DISABLED) 

            if hasattr(self, 'raw_model_mask_pil') and self.raw_model_mask_pil and \
               hasattr(self, 'lbl_raw_model_mask_canvas') and self.lbl_raw_model_mask_canvas.winfo_exists():
                self.on_generic_canvas_resize(None, self.lbl_raw_model_mask_canvas, 'raw_model_mask_pil')
            elif hasattr(self, 'lbl_raw_model_mask_canvas') and self.lbl_raw_model_mask_canvas.winfo_exists():
                self.lbl_raw_model_mask_canvas.delete("all") 
                if hasattr(self.lbl_raw_model_mask_canvas, "_current_tk_image_ref"):
                    try: delattr(self.lbl_raw_model_mask_canvas, "_current_tk_image_ref")
                    except: pass


            if hasattr(self, 'skeleton_model_scale_pil') and self.skeleton_model_scale_pil and \
               hasattr(self, 'lbl_skeleton_mask_canvas') and self.lbl_skeleton_mask_canvas.winfo_exists():
                self.on_generic_canvas_resize(None, self.lbl_skeleton_mask_canvas, 'skeleton_model_scale_pil')
            elif hasattr(self, 'lbl_skeleton_mask_canvas') and self.lbl_skeleton_mask_canvas.winfo_exists():
                self.lbl_skeleton_mask_canvas.delete("all") 
                if hasattr(self.lbl_skeleton_mask_canvas, "_current_tk_image_ref"):
                    try: delattr(self.lbl_skeleton_mask_canvas, "_current_tk_image_ref")
                    except: pass

        else: 
            if hasattr(self, 'lbl_image_status') and self.lbl_image_status.winfo_exists():
                self.lbl_image_status.config(text="Görüntü işlenemedi.")
            if hasattr(self, 'canvas_image') and self.canvas_image.winfo_exists():
                self.canvas_image.delete("all")
            self.original_cv_image = self.image_with_skeleton_overlay_for_selection = self.displayed_image_pil = self.mask_for_bfs_and_clicking_ORIG_SCALE = None
            self.current_image_source = None 
            if hasattr(self, 'lbl_raw_model_mask_canvas') and self.lbl_raw_model_mask_canvas.winfo_exists(): self.lbl_raw_model_mask_canvas.delete("all")
            if hasattr(self, 'lbl_skeleton_mask_canvas') and self.lbl_skeleton_mask_canvas.winfo_exists(): self.lbl_skeleton_mask_canvas.delete("all")


    def on_image_click(self, event):
        if self.is_camera_streaming_on_main_canvas:
            print("Kamera önizleme alanına tıklandı, görüntü yakalanıyor...")
            self.capture_frame_from_preview_and_process()
            return

        if self.image_with_skeleton_overlay_for_selection is None or \
           self.mask_for_bfs_and_clicking_ORIG_SCALE is None or \
           self.displayed_image_pil is None: 
            return 

        if not self.canvas_image.winfo_exists(): return

        canvas_w = self.canvas_image.winfo_width()
        canvas_h = self.canvas_image.winfo_height()
        if canvas_w <=1 or canvas_h <=1 or self.displayed_image_pil.width == 0 or self.displayed_image_pil.height == 0 : return


        img_disp_w = self.displayed_image_pil.width
        img_disp_h = self.displayed_image_pil.height

        offset_x = (canvas_w - img_disp_w) / 2
        offset_y = (canvas_h - img_disp_h) / 2

        if not (offset_x <= event.x < offset_x + img_disp_w and \
                offset_y <= event.y < offset_y + img_disp_h):
            return 

        click_x_on_displayed_img = event.x - offset_x
        click_y_on_displayed_img = event.y - offset_y

        orig_x = int((click_x_on_displayed_img / img_disp_w) * self.w_orig_for_path)
        orig_y = int((click_y_on_displayed_img / img_disp_h) * self.h_orig_for_path)

        orig_x = max(0, min(orig_x, self.w_orig_for_path - 1))
        orig_y = max(0, min(orig_y, self.h_orig_for_path - 1))


        if self.mask_for_bfs_and_clicking_ORIG_SCALE[orig_y, orig_x] != 1: 
            messagebox.showwarning("Geçersiz Nokta", "Lütfen labirentin vurgulanmış (muhtemel yol) çizgileri üzerine tıklayın.")
            return

        current_point_orig_coords = (orig_y, orig_x) 

        if self.start_point_original_coords is None:
            self.start_point_original_coords = current_point_orig_coords
            if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists():
                self.lbl_point_instruction.config(text="2. Bitiş noktasını seçin.")
        elif self.end_point_original_coords is None:
            if current_point_orig_coords == self.start_point_original_coords:
                messagebox.showinfo("Aynı Nokta", "Başlangıç ve bitiş noktaları aynı olamaz.")
                return
            self.end_point_original_coords = current_point_orig_coords
            if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists():
                self.lbl_point_instruction.config(text="Noktalar seçildi. 'Yolu Bul ve İşle' butonuna basabilirsiniz.")
            if hasattr(self, 'btn_process') and self.btn_process.winfo_exists():
                self.btn_process.config(state=tk.NORMAL) 
        else: 
            self.start_point_original_coords = current_point_orig_coords
            self.end_point_original_coords = None
            if hasattr(self, 'lbl_point_instruction') and self.lbl_point_instruction.winfo_exists():
                self.lbl_point_instruction.config(text="Başlangıç sıfırlandı. 2. Bitiş noktasını seçin.")
            if hasattr(self, 'btn_process') and self.btn_process.winfo_exists():
                self.btn_process.config(state=tk.DISABLED) 

        self.redraw_selected_points() 


    def process_maze(self):
        if not self.image_path or self.original_cv_image is None:
            messagebox.showerror("Eksik Bilgi", "Lütfen önce bir labirent görüntüsü yükleyin.")
            return
        if self.start_point_original_coords is None or self.end_point_original_coords is None:
            messagebox.showerror("Eksik Bilgi", "Lütfen başlangıç ve bitiş noktalarını seçin.")
            return
        if self.mask_for_bfs_and_clicking_ORIG_SCALE is None:
            messagebox.showerror("Hata", "Yol bulma için gerekli maske oluşturulamamış.")
            return
        if not self.current_image_source: 
            messagebox.showerror("Hata", "Görüntü kaynağı (galeri/kamera) belirlenmemiş. Lütfen tekrar görüntü yükleyin.")
            return


        self.start_progress()
        if hasattr(self, 'txt_commands') and self.txt_commands.winfo_exists():
            self.txt_commands.delete(1.0, tk.END)

        self.last_simplified_path_for_overlay = None
        self.last_generated_commands_for_pi_json = [] 
        self.last_generated_commands_for_display = []


        if hasattr(self, 'btn_animate_path') and self.btn_animate_path.winfo_exists(): self.btn_animate_path.config(state=tk.DISABLED)
        if hasattr(self, 'btn_animate_commands') and self.btn_animate_commands.winfo_exists(): self.btn_animate_commands.config(state=tk.DISABLED)
        if hasattr(self, 'btn_drive_vehicle') and self.btn_drive_vehicle.winfo_exists(): self.btn_drive_vehicle.config(state=tk.DISABLED)


        self.stop_path_animation() 
        self.stop_command_animation()

        output_image_final_display_cv = self.original_cv_image.copy() 

        path_found_pixels = self.find_path_bfs(
            self.mask_for_bfs_and_clicking_ORIG_SCALE, 
            self.start_point_original_coords,
            self.end_point_original_coords    
        )

        if path_found_pixels:
            simplified_path_nodes = self.simplify_path(path_found_pixels) 
            if simplified_path_nodes and len(simplified_path_nodes) >= 2:
                self.last_simplified_path_for_overlay = simplified_path_nodes

                self.last_generated_commands_for_pi_json, self.last_generated_commands_for_display = self.generate_and_process_commands(simplified_path_nodes)


                if self.last_generated_commands_for_display:
                    for cmd_text in self.last_generated_commands_for_display:
                        if hasattr(self, 'txt_commands') and self.txt_commands.winfo_exists():
                            self.txt_commands.insert(tk.END, cmd_text + "\n")
                    self.save_commands_to_file(self.last_generated_commands_for_display, "ui_directions.txt") 

                    if self.client_socket and hasattr(self, 'btn_drive_vehicle') and self.btn_drive_vehicle.winfo_exists():
                        self.btn_drive_vehicle.config(state=tk.NORMAL)
                else:
                    if hasattr(self, 'txt_commands') and self.txt_commands.winfo_exists():
                        self.txt_commands.insert(tk.END, "(Sonuçta komut listesi boş)\n")


                for i in range(len(simplified_path_nodes) - 1):
                    p1_cv = (simplified_path_nodes[i][1], simplified_path_nodes[i][0]) 
                    p2_cv = (simplified_path_nodes[i+1][1], simplified_path_nodes[i+1][0]) 
                    cv2.line(output_image_final_display_cv, p1_cv, p2_cv, LIVE_PATH_LINE_COLOR, LIVE_PATH_LINE_THICKNESS)

                start_node_cv = (simplified_path_nodes[0][1], simplified_path_nodes[0][0])
                end_node_cv = (simplified_path_nodes[-1][1], simplified_path_nodes[-1][0])
                cv2.circle(output_image_final_display_cv, start_node_cv, LIVE_POINT_RADIUS, LIVE_START_POINT_COLOR, LIVE_POINT_THICKNESS)
                cv2.circle(output_image_final_display_cv, end_node_cv, LIVE_POINT_RADIUS, LIVE_END_POINT_COLOR, LIVE_POINT_THICKNESS)

                self.result_image_display_tk_pil = Image.fromarray(cv2.cvtColor(output_image_final_display_cv, cv2.COLOR_BGR2RGB))
                if hasattr(self, 'lbl_result_image_canvas') and self.lbl_result_image_canvas.winfo_exists():
                    self.on_generic_canvas_resize_wrapper(None, self.lbl_result_image_canvas, 'result_image_display_tk_pil', 'current_animation_frame_pil')


                if hasattr(self, 'btn_animate_path') and self.btn_animate_path.winfo_exists(): self.btn_animate_path.config(state=tk.NORMAL)
                if self.last_generated_commands_for_display and hasattr(self, 'btn_animate_commands') and self.btn_animate_commands.winfo_exists():
                    self.btn_animate_commands.config(state=tk.NORMAL)
            else:
                messagebox.showinfo("Yol Bulunamadı", "Bulunan yol sadeleştirilemedi veya çok kısa.")
                self.result_image_display_tk_pil = None 
                if hasattr(self, 'lbl_result_image_canvas') and self.lbl_result_image_canvas.winfo_exists():
                    self.on_generic_canvas_resize_wrapper(None, self.lbl_result_image_canvas, 'result_image_display_tk_pil', 'current_animation_frame_pil')

        else:
            messagebox.showinfo("Yol Bulunamadı", f"BFS algoritması ile '{self.current_mask_type_str}' üzerinde yol bulunamadı.")
            self.result_image_display_tk_pil = None 
            if hasattr(self, 'lbl_result_image_canvas') and self.lbl_result_image_canvas.winfo_exists():
                self.on_generic_canvas_resize_wrapper(None, self.lbl_result_image_canvas, 'result_image_display_tk_pil', 'current_animation_frame_pil')


        if hasattr(self, 'notebook_results') and self.notebook_results.winfo_exists():
            self.notebook_results.select(self.tab_result_image) 

        if hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
            self.live_camera_status_label.config(text="Yol bulundu. Canlı kamera sekmesine geçilebilir." if (self.last_simplified_path_for_overlay or self.last_generated_commands_for_display) else "Canlı yol için önce bir yol bulunmalı.")
        self.stop_progress()

    def nullify_opposing_turns(self, commands_input):
        if not commands_input or len(commands_input) < 2:
            return commands_input

        standardized_commands = []
        for cmd in commands_input:
            if isinstance(cmd, str):
                parts = cmd.split()
                action = parts[0]
                value = int(parts[1]) if len(parts) > 1 and parts[0].startswith("ileri") else 0
                if parts[0] == "Sağ" and len(parts) > 1 and parts[1] == "Dön": action = "saga_don"
                elif parts[0] == "Sol" and len(parts) > 1 and parts[1] == "Dön": action = "sola_don"
                standardized_commands.append((action, value))
            elif isinstance(cmd, tuple) and len(cmd) == 2:
                standardized_commands.append(cmd)
            else: 
                standardized_commands.append(cmd)


        processed_commands_tuples = []
        i = 0
        n = len(standardized_commands)
        while i < n:
            if i + 1 < n:
                cmd1_action, cmd1_val = standardized_commands[i]
                cmd2_action, cmd2_val = standardized_commands[i+1]

                if (cmd1_action == "saga_don" and cmd2_action == "sola_don") or \
                   (cmd1_action == "sola_don" and cmd2_action == "saga_don"):
                    i += 2 
                else:
                    processed_commands_tuples.append(standardized_commands[i])
                    i += 1
            else:
                processed_commands_tuples.append(standardized_commands[i])
                i += 1
        return processed_commands_tuples 

    def _merge_short_forwards_across_turns(self, commands_input_tuples, min_acceptable_steps_for_merge_trigger):
        if not commands_input_tuples: return []
        processed_commands = []
        i = 0
        n = len(commands_input_tuples)
        while i < n:
            action, steps = commands_input_tuples[i]
            if action.startswith("ileri") and steps < min_acceptable_steps_for_merge_trigger:
                if i + 2 < n:
                    turn_action, _ = commands_input_tuples[i+1]
                    next_fwd_action, next_fwd_steps = commands_input_tuples[i+2]

                    is_turn_next = (turn_action == "saga_don" or turn_action == "sola_don")
                    is_forward_after_turn = next_fwd_action.startswith("ileri")

                    if is_turn_next and is_forward_after_turn:
                        
                        processed_commands.append(commands_input_tuples[i]) 
                        processed_commands.append(commands_input_tuples[i+1]) 
                        processed_commands.append(commands_input_tuples[i+2]) 
                        i += 3
                        continue
                    else:
                        processed_commands.append(commands_input_tuples[i])
                        i += 1
                else:
                    processed_commands.append(commands_input_tuples[i])
                    i += 1
            else:
                processed_commands.append(commands_input_tuples[i])
                i += 1
        return processed_commands


    def _simple_filter_short_forwards(self, commands_input_tuples, min_acceptable_steps):

        if not commands_input_tuples: return []
        filtered_command_list = []
        for action, value in commands_input_tuples:
            if action.startswith("ileri"):
                if value >= min_acceptable_steps:
                    filtered_command_list.append((action, value))
            else: 
                filtered_command_list.append((action, value))
        return filtered_command_list


    def generate_and_process_commands(self, simplified_path_nodes):
       

        raw_vehicle_commands_tuples = self.generate_vehicle_perspective_commands(simplified_path_nodes) 
        print(f"Ham komut (tuple) sayısı: {len(raw_vehicle_commands_tuples)}")

        processed_commands_tuples = list(raw_vehicle_commands_tuples)

        processed_commands_tuples = self.consolidate_vehicle_commands(processed_commands_tuples)
        print(f"Birleştirme 1 sonrası: {len(processed_commands_tuples)}")

        
        processed_commands_tuples = self._simple_filter_short_forwards(processed_commands_tuples, min_acceptable_steps=MIN_ACCEPTABLE_FORWARD_STEP)
        print(f"Kısa ileri filtreleme sonrası: {len(processed_commands_tuples)}")

        # Adım 6: Filtrelemeden sonra oluşabilecek ardışık aynı komutları tekrar birleştir
        processed_commands_tuples = self.consolidate_vehicle_commands(processed_commands_tuples)
        print(f"Birleştirme 3 (filtre sonrası) sonrası: {len(processed_commands_tuples)}")


        if processed_commands_tuples and (processed_commands_tuples[0][0] == "saga_don" or processed_commands_tuples[0][0] == "sola_don"):
           
            pass


       
        processed_commands_tuples = self.nullify_opposing_turns(processed_commands_tuples)
        print(f"Zıt dönüş iptali sonrası: {len(processed_commands_tuples)}")

        final_commands_tuples = self.consolidate_vehicle_commands(processed_commands_tuples)
        print(f"Birleştirme 4 (iptal sonrası) sonrası: {len(final_commands_tuples)}")


        if final_commands_tuples and (final_commands_tuples[-1][0] == "saga_don" or final_commands_tuples[-1][0] == "sola_don"):
            final_commands_tuples.pop()
            print(f"Son dönüş iptali sonrası: {len(final_commands_tuples)}")


        final_commands_for_display = []
        for action, value in final_commands_tuples:
            if action.startswith("ileri"): 
                direction_label = "Yatay" if action == "ileri_a" else "Dikey"
                final_commands_for_display.append(f"İleri ({direction_label}) {value}")
            elif action == "saga_don":
                final_commands_for_display.append("Sağ Dön")
            elif action == "sola_don":
                final_commands_for_display.append("Sol Dön")

        print(f"Nihai işlenmiş komut (tuple) sayısı: {len(final_commands_tuples)}.")
        return final_commands_tuples, final_commands_for_display


    def start_path_animation(self):
        if self.is_path_animating or self.is_command_animating:
            messagebox.showinfo("Animasyon Uyarısı", "Başka bir animasyon zaten çalışıyor.")
            return
        if not self.last_simplified_path_for_overlay or len(self.last_simplified_path_for_overlay) < 2:
            messagebox.showinfo("Animasyon Hatası", "Animasyon için geçerli bir yol bulunmuyor.")
            return
        if not self.original_vehicle_pil:
            messagebox.showerror("Animasyon Hatası", f"Araç görseli ('{VEHICLE_IMAGE_FILENAME}') yüklenemedi.")
            return
        if not self.result_image_display_tk_pil:
             messagebox.showerror("Animasyon Hatası", "Temel labirent görüntüsü (result_image_display_tk_pil) bulunmuyor.")
             return


        self.is_path_animating = True
        self.current_coord_path_segment_index = 0
        self.current_step_in_coord_segment = 0

        if hasattr(self, 'notebook_results') and self.notebook_results.winfo_exists():
            self.notebook_results.select(self.tab_result_image); 

        if hasattr(self, 'btn_animate_path') and self.btn_animate_path.winfo_exists(): self.btn_animate_path.config(state=tk.DISABLED);
        if hasattr(self, 'btn_animate_commands') and self.btn_animate_commands.winfo_exists(): self.btn_animate_commands.config(state=tk.DISABLED);
        if hasattr(self, 'btn_process') and self.btn_process.winfo_exists(): self.btn_process.config(state=tk.DISABLED)
        if hasattr(self, 'btn_drive_vehicle') and self.btn_drive_vehicle.winfo_exists(): self.btn_drive_vehicle.config(state=tk.DISABLED)


        self.animate_path_step()

    def stop_path_animation(self):
        if self.path_animation_job_id:
            try: self.root.after_cancel(self.path_animation_job_id)
            except tk.TclError: pass 
            self.path_animation_job_id = None
        self.is_path_animating = False

        if not self.is_command_animating:
            if self.last_simplified_path_for_overlay: 
                if hasattr(self, 'btn_animate_path') and self.btn_animate_path.winfo_exists(): self.btn_animate_path.config(state=tk.NORMAL)
                if self.last_generated_commands_for_display and hasattr(self, 'btn_animate_commands') and self.btn_animate_commands.winfo_exists():
                     self.btn_animate_commands.config(state=tk.NORMAL)
            if hasattr(self, 'btn_process') and self.btn_process.winfo_exists():
                self.btn_process.config(state=tk.NORMAL if self.start_point_original_coords and self.end_point_original_coords else tk.DISABLED)
            if hasattr(self, 'btn_drive_vehicle') and self.btn_drive_vehicle.winfo_exists() and self.client_socket and self.last_generated_commands_for_pi_json:
                self.btn_drive_vehicle.config(state=tk.NORMAL)


        if hasattr(self, 'result_image_display_tk_pil') and self.result_image_display_tk_pil and not (self.is_path_animating or self.is_command_animating):
            if hasattr(self, 'lbl_result_image_canvas') and self.lbl_result_image_canvas.winfo_exists():
                self.on_generic_canvas_resize_wrapper(None, self.lbl_result_image_canvas, 'result_image_display_tk_pil', 'current_animation_frame_pil')


    def animate_path_step(self):
        if not self.is_path_animating or not self.root.winfo_exists():
            self.stop_path_animation()
            return

        path = self.last_simplified_path_for_overlay
        if not path or self.current_coord_path_segment_index >= len(path) - 1: 
            self.stop_path_animation()
            return

        p1_orig = path[self.current_coord_path_segment_index] 
        p2_orig = path[self.current_coord_path_segment_index + 1] 

        x1_orig, y1_orig = p1_orig[1], p1_orig[0]
        x2_orig, y2_orig = p2_orig[1], p2_orig[0]

        t = self.current_step_in_coord_segment / STEPS_PER_SEGMENT
        current_x_orig = x1_orig + t * (x2_orig - x1_orig)
        current_y_orig = y1_orig + t * (y2_orig - y1_orig)

        dx = x2_orig - x1_orig
        dy = y2_orig - y1_orig
        angle_rad = math.atan2(dy, dx) 
        angle_deg = -math.degrees(angle_rad) 

        if not self.original_vehicle_pil or not self.result_image_display_tk_pil:
            self.stop_path_animation()
            return

        rotated_vehicle_pil = self.original_vehicle_pil.rotate(angle_deg, expand=True, resample=Image.Resampling.BICUBIC)

        composite_image_pil = self.result_image_display_tk_pil.copy()

        paste_x = int(current_x_orig - rotated_vehicle_pil.width / 2)
        paste_y = int(current_y_orig - rotated_vehicle_pil.height / 2)

        if rotated_vehicle_pil.mode == 'RGBA':
            composite_image_pil.paste(rotated_vehicle_pil, (paste_x, paste_y), rotated_vehicle_pil)
        else:
            composite_image_pil.paste(rotated_vehicle_pil, (paste_x, paste_y))

        self.current_animation_frame_pil = composite_image_pil 

        if hasattr(self, 'lbl_result_image_canvas') and self.lbl_result_image_canvas.winfo_exists():
             self.on_generic_canvas_resize_wrapper(None, self.lbl_result_image_canvas, 'result_image_display_tk_pil', 'current_animation_frame_pil')


        self.current_step_in_coord_segment += 1
        if self.current_step_in_coord_segment >= STEPS_PER_SEGMENT:
            self.current_step_in_coord_segment = 0
            self.current_coord_path_segment_index += 1

        if self.is_path_animating:
            self.path_animation_job_id = self.root.after(ANIMATION_DELAY_MS, self.animate_path_step)


    def start_command_animation(self):
        if self.is_command_animating or self.is_path_animating:
            messagebox.showinfo("Animasyon Uyarısı", "Başka bir animasyon zaten çalışıyor.")
            return
        if not self.last_generated_commands_for_display: 
            messagebox.showinfo("Animasyon Hatası", "Animasyon için işlenmiş komut bulunmuyor.")
            return
        if not self.original_vehicle_pil:
            messagebox.showerror("Animasyon Hatası", f"Araç görseli ('{VEHICLE_IMAGE_FILENAME}') yüklenemedi.")
            return
        if not self.result_image_display_tk_pil:
             messagebox.showerror("Animasyon Hatası", "Temel labirent görüntüsü bulunmuyor.")
             return
        if not self.last_simplified_path_for_overlay or len(self.last_simplified_path_for_overlay) < 2:
            messagebox.showerror("Animasyon Hatası", "Başlangıç pozisyonu ve yönü için yol bilgisi eksik.")
            return


        self.is_command_animating = True
        self.current_command_index_for_anim = 0
        self.steps_taken_for_current_fwd_cmd_anim = 0
        self.total_steps_for_current_fwd_cmd_anim = 0

        start_node_y, start_node_x = self.last_simplified_path_for_overlay[0]
        self.vehicle_pos_x_cmd_anim = float(start_node_x)
        self.vehicle_pos_y_cmd_anim = float(start_node_y)

        p0_y, p0_x = self.last_simplified_path_for_overlay[0]
        p1_y, p1_x = self.last_simplified_path_for_overlay[1]
        initial_dy_map = p1_y - p0_y
        initial_dx_map = p1_x - p0_x

        if initial_dy_map != 0: 
            self.vehicle_orientation_dy_anim = int(np.sign(initial_dy_map))
            self.vehicle_orientation_dx_anim = 0
        elif initial_dx_map != 0: 
            self.vehicle_orientation_dy_anim = 0
            self.vehicle_orientation_dx_anim = int(np.sign(initial_dx_map))
        else: 
            self.vehicle_orientation_dy_anim = 0
            self.vehicle_orientation_dx_anim = 1


        if hasattr(self, 'notebook_results') and self.notebook_results.winfo_exists():
            self.notebook_results.select(self.tab_result_image);

        if hasattr(self, 'btn_animate_path') and self.btn_animate_path.winfo_exists(): self.btn_animate_path.config(state=tk.DISABLED);
        if hasattr(self, 'btn_animate_commands') and self.btn_animate_commands.winfo_exists(): self.btn_animate_commands.config(state=tk.DISABLED);
        if hasattr(self, 'btn_process') and self.btn_process.winfo_exists(): self.btn_process.config(state=tk.DISABLED)
        if hasattr(self, 'btn_drive_vehicle') and self.btn_drive_vehicle.winfo_exists(): self.btn_drive_vehicle.config(state=tk.DISABLED)


        self.animate_command_step()

    def stop_command_animation(self):
        if self.command_animation_job_id:
            try: self.root.after_cancel(self.command_animation_job_id)
            except tk.TclError: pass
            self.command_animation_job_id = None
        self.is_command_animating = False

        if not self.is_path_animating: 
            if self.last_simplified_path_for_overlay:
                if hasattr(self, 'btn_animate_path') and self.btn_animate_path.winfo_exists(): self.btn_animate_path.config(state=tk.NORMAL)
                if self.last_generated_commands_for_display and hasattr(self, 'btn_animate_commands') and self.btn_animate_commands.winfo_exists():
                     self.btn_animate_commands.config(state=tk.NORMAL)
            if hasattr(self, 'btn_process') and self.btn_process.winfo_exists():
                self.btn_process.config(state=tk.NORMAL if self.start_point_original_coords and self.end_point_original_coords else tk.DISABLED)
            if hasattr(self, 'btn_drive_vehicle') and self.btn_drive_vehicle.winfo_exists() and self.client_socket and self.last_generated_commands_for_pi_json:
                self.btn_drive_vehicle.config(state=tk.NORMAL)

        if hasattr(self, 'result_image_display_tk_pil') and self.result_image_display_tk_pil and not (self.is_path_animating or self.is_command_animating):
            if hasattr(self, 'lbl_result_image_canvas') and self.lbl_result_image_canvas.winfo_exists():
                self.on_generic_canvas_resize_wrapper(None, self.lbl_result_image_canvas, 'result_image_display_tk_pil', 'current_animation_frame_pil')


    def animate_command_step(self):
        if not self.is_command_animating or not self.root.winfo_exists():
            self.stop_command_animation()
            return

        if self.current_command_index_for_anim >= len(self.last_generated_commands_for_display):
            self.stop_command_animation()
            return

        command_str_for_anim = self.last_generated_commands_for_display[self.current_command_index_for_anim]

        if command_str_for_anim.startswith("İleri"): 
            if self.steps_taken_for_current_fwd_cmd_anim == 0: 
                try:
                    
                    self.total_steps_for_current_fwd_cmd_anim = int(command_str_for_anim.split()[-1])
                except (ValueError, IndexError): 
                    self.current_command_index_for_anim += 1
                    self.command_animation_job_id = self.root.after(ANIMATION_DELAY_MS, self.animate_command_step)
                    return

            if self.steps_taken_for_current_fwd_cmd_anim < self.total_steps_for_current_fwd_cmd_anim:
                move_amount = min(PIXEL_MOVE_PER_COMMAND_ANIM_STEP, self.total_steps_for_current_fwd_cmd_anim - self.steps_taken_for_current_fwd_cmd_anim)
                self.vehicle_pos_x_cmd_anim += self.vehicle_orientation_dx_anim * move_amount
                self.vehicle_pos_y_cmd_anim += self.vehicle_orientation_dy_anim * move_amount
                self.steps_taken_for_current_fwd_cmd_anim += move_amount

            if self.steps_taken_for_current_fwd_cmd_anim >= self.total_steps_for_current_fwd_cmd_anim:
                self.steps_taken_for_current_fwd_cmd_anim = 0 
                self.total_steps_for_current_fwd_cmd_anim = 0
                self.current_command_index_for_anim += 1

        elif command_str_for_anim == "Sağ Dön":
            new_dy = self.vehicle_orientation_dx_anim
            new_dx = -self.vehicle_orientation_dy_anim
            self.vehicle_orientation_dy_anim = new_dy
            self.vehicle_orientation_dx_anim = new_dx
            self.current_command_index_for_anim += 1
        elif command_str_for_anim == "Sol Dön":
            new_dy = -self.vehicle_orientation_dx_anim
            new_dx = self.vehicle_orientation_dy_anim
            self.vehicle_orientation_dy_anim = new_dy
            self.vehicle_orientation_dx_anim = new_dx
            self.current_command_index_for_anim += 1
        else:
            self.current_command_index_for_anim += 1


        if not self.result_image_display_tk_pil or not self.original_vehicle_pil:
            self.stop_command_animation()
            return

        composite_image_pil = self.result_image_display_tk_pil.copy() 

        angle_for_pil_rad = math.atan2(self.vehicle_orientation_dy_anim, self.vehicle_orientation_dx_anim)
        angle_for_pil_deg = -math.degrees(angle_for_pil_rad)

        rotated_vehicle_pil = self.original_vehicle_pil.rotate(angle_for_pil_deg, expand=True, resample=Image.Resampling.BICUBIC)

        paste_x = int(self.vehicle_pos_x_cmd_anim - rotated_vehicle_pil.width / 2)
        paste_y = int(self.vehicle_pos_y_cmd_anim - rotated_vehicle_pil.height / 2)

        if rotated_vehicle_pil.mode == 'RGBA':
            composite_image_pil.paste(rotated_vehicle_pil, (paste_x, paste_y), rotated_vehicle_pil)
        else:
            composite_image_pil.paste(rotated_vehicle_pil, (paste_x, paste_y))

        self.current_animation_frame_pil = composite_image_pil
        if hasattr(self, 'lbl_result_image_canvas') and self.lbl_result_image_canvas.winfo_exists():
            self.on_generic_canvas_resize_wrapper(None, self.lbl_result_image_canvas, 'result_image_display_tk_pil', 'current_animation_frame_pil')


        if self.is_command_animating:
            self.command_animation_job_id = self.root.after(ANIMATION_DELAY_MS, self.animate_command_step)


    def find_path_bfs(self, grid, start_node, end_node):
        if grid.ndim != 2: return None
        rows, cols = grid.shape

        if not (0 <= start_node[0] < rows and 0 <= start_node[1] < cols and grid[start_node[0], start_node[1]] == 1):
            print(f"BFS: Başlangıç noktası geçersiz: {start_node}, grid değeri: {grid[start_node[0], start_node[1]] if (0 <= start_node[0] < rows and 0 <= start_node[1] < cols) else 'sınır dışı'}")
            return None
        if not (0 <= end_node[0] < rows and 0 <= end_node[1] < cols and grid[end_node[0], end_node[1]] == 1):
            print(f"BFS: Bitiş noktası geçersiz: {end_node}, grid değeri: {grid[end_node[0], end_node[1]] if (0 <= end_node[0] < rows and 0 <= end_node[1] < cols) else 'sınır dışı'}")
            return None

        queue = deque([(start_node, [start_node])]) 
        visited_nodes = {start_node}

        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1,1), (1,-1), (-1,1), (-1,-1)] 

        while queue:
            (current_node, current_path) = queue.popleft()
            if current_node == end_node:
                return current_path 

            for dr, dc in possible_moves:
                next_r, next_c = current_node[0] + dr, current_node[1] + dc
                neighbor_node = (next_r, next_c)

                if 0 <= next_r < rows and 0 <= next_c < cols and \
                   grid[next_r, next_c] == 1 and neighbor_node not in visited_nodes: 
                    visited_nodes.add(neighbor_node)
                    new_path = list(current_path)
                    new_path.append(neighbor_node)
                    queue.append((neighbor_node, new_path))
        return None 


    def simplify_path(self, path_input_pixels): 
        if not path_input_pixels or len(path_input_pixels) < 2:
            return path_input_pixels

        simplified_nodes = [path_input_pixels[0]] 
        for i in range(1, len(path_input_pixels) - 1):
            dy1 = path_input_pixels[i][0] - path_input_pixels[i-1][0]
            dx1 = path_input_pixels[i][1] - path_input_pixels[i-1][1]
            dy2 = path_input_pixels[i+1][0] - path_input_pixels[i][0]
            dx2 = path_input_pixels[i+1][1] - path_input_pixels[i][1]

            dir1 = (np.sign(dy1), np.sign(dx1))
            dir2 = (np.sign(dy2), np.sign(dx2))

            if dir1 != dir2:
                if simplified_nodes[-1] != path_input_pixels[i]: 
                     simplified_nodes.append(path_input_pixels[i])

        if simplified_nodes[-1] != path_input_pixels[-1]:
            simplified_nodes.append(path_input_pixels[-1])


        return simplified_nodes


    def _calculate_turns(self, current_dy, current_dx, target_dy, target_dx):
        
        turn_sequence = []
        if (current_dy, current_dx) == (target_dy, target_dx):
            return turn_sequence 

      
        temp_dy_right, temp_dx_right = current_dx, -current_dy
        if (temp_dy_right, temp_dx_right) == (target_dy, target_dx):
            turn_sequence.append("saga_don")
            return turn_sequence

        temp_dy_left, temp_dx_left = -current_dx, current_dy
        if (temp_dy_left, temp_dx_left) == (target_dy, target_dx):
            turn_sequence.append("sola_don")
            return turn_sequence

       
        if (-current_dy, -current_dx) == (target_dy, target_dx):
            turn_sequence.extend(["saga_don", "saga_don"]) 
            return turn_sequence

       
        print(f"UYARI: Dönüş hesaplanamadı! Mevcut: ({current_dy},{current_dx}), Hedef: ({target_dy},{target_dx})")
        return []


    def generate_vehicle_perspective_commands(self, simplified_path_nodes):
        if not simplified_path_nodes or len(simplified_path_nodes) < 2:
            return []

        vehicle_commands_tuples = []
        current_orientation_dy, current_orientation_dx = (0, 0)

        p0_y, p0_x = simplified_path_nodes[0]
        p1_y, p1_x = simplified_path_nodes[1]

        initial_delta_y = p1_y - p0_y
        initial_delta_x = p1_x - p0_x
        initial_forward_steps = 0
        initial_forward_command_type = "" 

        if initial_delta_y != 0: 
            initial_forward_steps = abs(initial_delta_y)
            current_orientation_dy = np.sign(initial_delta_y)
            current_orientation_dx = 0
            initial_forward_command_type = "ileri_b"
        elif initial_delta_x != 0: 
            initial_forward_steps = abs(initial_delta_x)
            current_orientation_dx = np.sign(initial_delta_x)
            current_orientation_dy = 0
            initial_forward_command_type = "ileri_a"
        else: 
            return []

        if initial_forward_steps > 0:
            vehicle_commands_tuples.append((initial_forward_command_type, initial_forward_steps))

        for i in range(1, len(simplified_path_nodes) - 1):
            seg_start_y, seg_start_x = simplified_path_nodes[i]
            seg_end_y, seg_end_x = simplified_path_nodes[i+1]

            target_seg_delta_y = seg_end_y - seg_start_y
            target_seg_delta_x = seg_end_x - seg_start_x

            if target_seg_delta_y == 0 and target_seg_delta_x == 0: continue 
            next_segment_intended_orientation_dy, next_segment_intended_orientation_dx = (0,0)
            current_segment_interpreted_forward_steps = 0
            current_segment_forward_command_type = ""

            if target_seg_delta_y != 0:
                current_segment_interpreted_forward_steps = abs(target_seg_delta_y)
                next_segment_intended_orientation_dy = np.sign(target_seg_delta_y)
                next_segment_intended_orientation_dx = 0
                current_segment_forward_command_type = "ileri_b"
            elif target_seg_delta_x != 0:
                current_segment_interpreted_forward_steps = abs(target_seg_delta_x)
                next_segment_intended_orientation_dx = np.sign(target_seg_delta_x)
                next_segment_intended_orientation_dy = 0
                current_segment_forward_command_type = "ileri_a"

            if current_segment_interpreted_forward_steps == 0: continue 

            turns = self._calculate_turns(current_orientation_dy, current_orientation_dx,
                                          next_segment_intended_orientation_dy, next_segment_intended_orientation_dx)
            if turns:
                for turn_action_str in turns: 
                    vehicle_commands_tuples.append((turn_action_str, 0)) 
                current_orientation_dy = next_segment_intended_orientation_dy
                current_orientation_dx = next_segment_intended_orientation_dx

            if current_segment_interpreted_forward_steps > 0:
                vehicle_commands_tuples.append((current_segment_forward_command_type, current_segment_interpreted_forward_steps))

        return vehicle_commands_tuples


    def consolidate_vehicle_commands(self, commands_input_tuples):
       
        if not commands_input_tuples:
            return []
        consolidated_list = []
        i = 0
        while i < len(commands_input_tuples):
            current_action, current_value = commands_input_tuples[i]
            if current_action.startswith("ileri"): 
                total_steps = current_value
                j = i + 1
                while j < len(commands_input_tuples) and commands_input_tuples[j][0] == current_action:
                    total_steps += commands_input_tuples[j][1]
                    j += 1
                consolidated_list.append((current_action, total_steps))
                i = j 
            else:
                consolidated_list.append((current_action, current_value))
                i += 1
        return consolidated_list

    def merge_small_vehicle_steps(self, commands_input_tuples, threshold=SMALL_STEP_THRESHOLD):
      
        return commands_input_tuples 


    def save_commands_to_file(self, commands_list_to_save, filename="direction.txt"):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for command_item_str in commands_list_to_save:
                    f.write(command_item_str + "\n")
            print(f"Komutlar başarıyla '{filename}' dosyasına kaydedildi.")
        except IOError as e_io:
            messagebox.showerror("Dosya Kaydetme Hatası", f"Komutlar '{filename}' dosyasına kaydedilemedi:\n{e_io}")
        except Exception as e_gen:
            messagebox.showerror("Dosya Kaydetme Hatası", f"Beklenmedik bir hata oluştu:\n{e_gen}")


    def start_live_camera_feed(self):
        if self.is_live_camera_active: return

        if not self.last_simplified_path_for_overlay and not self.last_generated_commands_for_display:
            if hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
                self.live_camera_status_label.config(text="Canlı yol gösterimi için önce 'Yolu Bul ve İşle' yapılmalı.")
            return

        if hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
            self.live_camera_status_label.config(text="Kamera başlatılıyor...")


        self.is_live_camera_active = True
        self.stop_camera_event.clear() 

        if self.video_capture_device is None or not self.video_capture_device.isOpened():
            self.video_capture_device = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
            if not (self.video_capture_device and self.video_capture_device.isOpened()):
                self.video_capture_device = cv2.VideoCapture(1, cv2.CAP_DSHOW)

            if not (self.video_capture_device and self.video_capture_device.isOpened()):
                messagebox.showerror("Kamera Hatası", "Canlı yayın için kamera açılamadı.")
                self.is_live_camera_active = False
                self.video_capture_device = None
                if hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
                    self.live_camera_status_label.config(text="Kamera açılamadı.")
                return

        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        if hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
            self.live_camera_status_label.config(text="Canlı kamera aktif.")


    def stop_live_camera_feed(self):
        if self.is_live_camera_active:
            print("Canlı kamera yayını durduruluyor (sonuçlar sekmesi)...")
            self.stop_camera_event.set() 
            if self.camera_thread and self.camera_thread.is_alive():
                print("Kamera thread'inin sonlanması bekleniyor (sonuçlar sekmesi)...")
                self.camera_thread.join(timeout=1.0) 
                if self.camera_thread.is_alive():
                    print("UYARI: Kamera thread'i (sonuçlar sekmesi) zamanında sonlanmadı.")


            self.is_live_camera_active = False 
            if self.video_capture_device and self.video_capture_device.isOpened():
                print("Video yakalama cihazı serbest bırakılıyor (stop_live_camera_feed - sonuçlar sekmesi).")
                self.video_capture_device.release()
            self.video_capture_device = None 

            if self.root.winfo_exists() and hasattr(self, 'live_camera_canvas') and self.live_camera_canvas.winfo_exists():
                self.root.after(0, lambda: self.live_camera_canvas.delete("all") if self.live_camera_canvas.winfo_exists() else None)
                if hasattr(self.live_camera_canvas, "_current_tk_image_ref"): 
                    try: delattr(self.live_camera_canvas, "_current_tk_image_ref")
                    except: pass


            if hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
                self.live_camera_status_label.config(text="Canlı kamera durduruldu.")
            print("Canlı kamera yayını (sonuçlar sekmesi) başarıyla durduruldu.")


    def _camera_loop(self):
        try:
            if self.video_capture_device is None or not self.video_capture_device.isOpened():
                if self.root.winfo_exists() and hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
                    self.root.after(0, lambda: self.live_camera_status_label.config(text="Kamera hatası!"))
                self.is_live_camera_active = False
                return

            use_scaling_for_path_overlay = not (self.h_orig_for_path == 0 or self.w_orig_for_path == 0)

            while not self.stop_camera_event.is_set():
                if not (self.video_capture_device and self.video_capture_device.isOpened()):
                    break 
                ret, frame = self.video_capture_device.read()
                if not ret:
                    cv2.waitKey(100) 
                    continue

                frame_with_path_overlay = frame.copy()
                current_frame_h, current_frame_w = frame_with_path_overlay.shape[:2]

                scale_x_live_cam, scale_y_live_cam = 1.0, 1.0
                if use_scaling_for_path_overlay and self.w_orig_for_path > 0 and self.h_orig_for_path > 0 and \
                   current_frame_w > 0 and current_frame_h > 0:
                    scale_x_live_cam = current_frame_w / self.w_orig_for_path
                    scale_y_live_cam = current_frame_h / self.h_orig_for_path


                if self.last_generated_commands_for_display and self.last_simplified_path_for_overlay and len(self.last_simplified_path_for_overlay) >=2:
                    draw_y_orig, draw_x_orig = self.last_simplified_path_for_overlay[0] # Başlangıç noktası (y,x)

                    p0_y_orig, p0_x_orig = self.last_simplified_path_for_overlay[0]
                    p1_y_orig, p1_x_orig = self.last_simplified_path_for_overlay[1]
                    initial_delta_y = p1_y_orig - p0_y_orig
                    initial_delta_x = p1_x_orig - p0_x_orig
                    current_orientation_dy_live, current_orientation_dx_live = 0,0
                    if initial_delta_y != 0: current_orientation_dy_live = int(np.sign(initial_delta_y)); current_orientation_dx_live = 0
                    elif initial_delta_x != 0: current_orientation_dx_live = int(np.sign(initial_delta_x)); current_orientation_dy_live = 0
                    else: current_orientation_dx_live = 1 

                    last_scaled_x = int(draw_x_orig * scale_x_live_cam)
                    last_scaled_y = int(draw_y_orig * scale_y_live_cam)
                    cv2.circle(frame_with_path_overlay, (last_scaled_x, last_scaled_y), LIVE_POINT_RADIUS, LIVE_START_POINT_COLOR, LIVE_POINT_THICKNESS)

                    for command_str_live in self.last_generated_commands_for_display:
                        if command_str_live.startswith("İleri"): 
                            try:
                                steps = int(command_str_live.split()[-1])
                                draw_x_orig += current_orientation_dx_live * steps
                                draw_y_orig += current_orientation_dy_live * steps
                                current_scaled_x = int(draw_x_orig * scale_x_live_cam)
                                current_scaled_y = int(draw_y_orig * scale_y_live_cam)
                                cv2.line(frame_with_path_overlay, (last_scaled_x, last_scaled_y), (current_scaled_x, current_scaled_y), COMMAND_PATH_LINE_COLOR, COMMAND_PATH_LINE_THICKNESS)
                                last_scaled_x, last_scaled_y = current_scaled_x, current_scaled_y
                            except: continue
                        elif command_str_live == "Sağ Dön":
                            cv2.circle(frame_with_path_overlay, (last_scaled_x, last_scaled_y), TURN_MARKER_RADIUS, TURN_MARKER_COLOR, TURN_MARKER_THICKNESS)
                            new_o_dy = current_orientation_dx_live; new_o_dx = -current_orientation_dy_live
                            current_orientation_dy_live, current_orientation_dx_live = new_o_dy, new_o_dx
                        elif command_str_live == "Sol Dön":
                            cv2.circle(frame_with_path_overlay, (last_scaled_x, last_scaled_y), TURN_MARKER_RADIUS, TURN_MARKER_COLOR, TURN_MARKER_THICKNESS)
                            new_o_dy = -current_orientation_dx_live; new_o_dx = current_orientation_dy_live
                            current_orientation_dy_live, current_orientation_dx_live = new_o_dy, new_o_dx
                    cv2.circle(frame_with_path_overlay, (last_scaled_x, last_scaled_y), LIVE_POINT_RADIUS, LIVE_END_POINT_COLOR, LIVE_POINT_THICKNESS)

                elif self.last_simplified_path_for_overlay: 
                    path_nodes_orig_coords = self.last_simplified_path_for_overlay
                    for i in range(len(path_nodes_orig_coords) - 1):
                        p1_orig_y, p1_orig_x = path_nodes_orig_coords[i]
                        p2_orig_y, p2_orig_x = path_nodes_orig_coords[i+1]
                        p1_on_cam_frame = (int(p1_orig_x * scale_x_live_cam), int(p1_orig_y * scale_y_live_cam))
                        p2_on_cam_frame = (int(p2_orig_x * scale_x_live_cam), int(p2_orig_y * scale_y_live_cam))
                        cv2.line(frame_with_path_overlay, p1_on_cam_frame, p2_on_cam_frame, LIVE_PATH_LINE_COLOR, LIVE_PATH_LINE_THICKNESS)

                    if len(path_nodes_orig_coords) > 0:
                        start_node_y, start_node_x = path_nodes_orig_coords[0]
                        start_point_on_cam = (int(start_node_x * scale_x_live_cam), int(start_node_y * scale_y_live_cam))
                        cv2.circle(frame_with_path_overlay, start_point_on_cam, LIVE_POINT_RADIUS, LIVE_START_POINT_COLOR, LIVE_POINT_THICKNESS)
                        if len(path_nodes_orig_coords) > 1:
                            end_node_y, end_node_x = path_nodes_orig_coords[-1]
                            end_point_on_cam = (int(end_node_x * scale_x_live_cam), int(end_node_y * scale_y_live_cam))
                            cv2.circle(frame_with_path_overlay, end_point_on_cam, LIVE_POINT_RADIUS, LIVE_END_POINT_COLOR, LIVE_POINT_THICKNESS)


                img_rgb_live = cv2.cvtColor(frame_with_path_overlay, cv2.COLOR_BGR2RGB)
                pil_image_live_cam = Image.fromarray(img_rgb_live)

                def update_live_camera_canvas_on_ui_thread(pil_img_to_show):
                    if not self.stop_camera_event.is_set() and self.root.winfo_exists() and \
                       hasattr(self, 'live_camera_canvas') and self.live_camera_canvas.winfo_exists():
                        try:
                            canvas_w_live = self.live_camera_canvas.winfo_width()
                            canvas_h_live = self.live_camera_canvas.winfo_height()
                            if canvas_w_live > 1 and canvas_h_live > 1:
                                pil_img_to_show.thumbnail((canvas_w_live - 2, canvas_h_live - 2), Image.Resampling.LANCZOS)

                            self.live_camera_feed_tk = ImageTk.PhotoImage(image=pil_img_to_show)
                            setattr(self.live_camera_canvas, "_current_tk_image_ref", self.live_camera_feed_tk)
                            self.live_camera_canvas.delete("all")
                            self.live_camera_canvas.create_image(canvas_w_live // 2, canvas_h_live // 2, anchor=tk.CENTER, image=self.live_camera_feed_tk)
                        except Exception as e_ui_update:
                            if "application has been destroyed" in str(e_ui_update).lower() or \
                               "invalid command name" in str(e_ui_update).lower():
                                self.stop_camera_event.set() 


                if self.root.winfo_exists():
                    self.root.after(0, lambda img=pil_image_live_cam: update_live_camera_canvas_on_ui_thread(img))
                else:
                    break

                self.stop_camera_event.wait(timeout=(ANIMATION_DELAY_MS / 1000.0) / 2.0 ) 
        except Exception as e_cam_loop:
            print(f"Kamera döngüsü HATA (sonuçlar sekmesi): {e_cam_loop}")
            traceback.print_exc()
        finally:
            self.is_live_camera_active = False
            if self.root.winfo_exists() and hasattr(self, 'live_camera_status_label') and self.live_camera_status_label.winfo_exists():
                self.root.after(0, lambda: self.live_camera_status_label.config(text="Canlı kamera durdu."))


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: Model dosyası '{MODEL_PATH}' bulunamadı. Lütfen MODEL_PATH değişkenini güncelleyin.")
        try:
            print("UYARI: Gerçek model bulunamadı. Test için dummy model oluşturuluyor (TensorFlow gerektirir).")
            dummy_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
                tf.keras.layers.Conv2D(1, (3,3), padding='same', activation='sigmoid')
            ])
            dummy_model.save(MODEL_PATH)
            print(f"Dummy model '{MODEL_PATH}' oluşturuldu.")
        except Exception as e_dummy:
            print(f"Dummy model oluşturulamadı: {e_dummy}. Model yükleme başarısız olacak.")


    if not os.path.exists(VEHICLE_IMAGE_FILENAME):
        print(f"UYARI: Araç görseli '{VEHICLE_IMAGE_FILENAME}' bulunamadı. Animasyonlar araçsız olabilir.")
        try:
            dummy_vehicle = Image.new('RGBA', (50,30), (255,0,0,128)) 
            dummy_vehicle.save(VEHICLE_IMAGE_FILENAME)
            print(f"Dummy araç görseli '{VEHICLE_IMAGE_FILENAME}' oluşturuldu.")
        except Exception as e_dummy_vehicle:
            print(f"Dummy araç görseli oluşturulamadı: {e_dummy_vehicle}")


    root = tk.Tk()
    app = MazeSolverApp(root)
    root.mainloop()
