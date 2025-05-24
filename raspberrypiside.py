import RPi.GPIO as GPIO
import curses
import time
import smbus
import math
import socket
import json # Komutları JSON formatında almak için
import traceback

# ============ MPU6050 Ayarları ============ #
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
GYRO_XOUT_H = 0x43 # Jiroskop X ekseni yüksek byte adresi

bus = None
mpu_initialized = False

# ============ MOTOR Ayarları ve PWM ============ #
M1_IN1 = 5
M1_IN2 = 6
M2_IN3 = 13
M2_IN4 = 19
M3_IN1 = 18
M3_IN2 = 23
M4_IN3 = 24
M4_IN4 = 25

PWM_M1_PIN = 20
PWM_M2_PIN = 21
PWM_M3_PIN = 16
PWM_M4_PIN = 12

EFFECTIVE_MIN_PWM = 35
MIN_SPEED = EFFECTIVE_MIN_PWM
MAX_SPEED = 85
PWM_DEADBAND = 8

FORWARD_SPEED = max(30, EFFECTIVE_MIN_PWM) # Genel ileri hareket hızı
# BACKWARD_SPEED = max(30, EFFECTIVE_MIN_PWM) # Geri hareket şu an kullanılmıyor

# Farklı görüntü kaynakları ve hareket yönleri için adım kazançları
ADIM_KAZANCI_CAMERA_A = 0.068  # Kamera için yatay (ileri_a) adım kazancı
ADIM_KAZANCI_CAMERA_B = 0.08  # Kamera için dikey (ileri_b) adım kazancı
ADIM_KAZANCI_GALLERY_A = 0.2 # Galeri için yatay (ileri_a) adım kazancı
ADIM_KAZANCI_GALLERY_B = 0.25 # Galeri için dikey (ileri_b) adım kazancı

# Başlangıçta kullanılacak aktif kazançlar (CALIBRATE ile güncellenecek)
current_adim_kazanci_a = ADIM_KAZANCI_CAMERA_A
current_adim_kazanci_b = ADIM_KAZANCI_CAMERA_B

FORWARD_DURATION_PER_STEP = 0.1 # Bu, ADIM_KAZANCI ile çarpılan adıma karşılık gelen süre
DELAY_BETWEEN_COMMANDS = 0.2
LED_PIN = 26

SERVER_HOST = '192.168.137.1' # PC'nizin IP adresini buraya girin (veya 0.0.0.0)
SERVER_PORT = 65432
client_socket = None
connected_to_server = False

pwm_m1, pwm_m2, pwm_m3, pwm_m4 = None, None, None, None
pwm_led = None
pwm_initialized = False
led_pwm_initialized = False

direction_pins = [M1_IN1, M1_IN2, M2_IN3, M2_IN4, M3_IN1, M3_IN2, M4_IN3, M4_IN4]
pwm_hardware_pins = [PWM_M1_PIN, PWM_M2_PIN, PWM_M3_PIN, PWM_M4_PIN]
offset_x = 0.0 # turn_pid içinde kullanılacak global kalibrasyon ofseti
current_image_source_on_pi = "CAMERA" # PC'den gelen görüntü kaynağını saklamak için (varsayılan)

def initialize_mpu(stdscr):
    global bus, mpu_initialized
    try:
        bus = smbus.SMBus(1)
        bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)
        stdscr.addstr(0, 0, "MPU6050 başarıyla başlatıldı.                                    ")
        mpu_initialized = True
    except Exception as e:
        stdscr.addstr(0, 0, f"MPU6050 başlatılırken hata: {e}")
        stdscr.addstr(1, 0, "Lütfen MPU6050'nin doğru bağlandığından ve I2C'nin etkin olduğundan emin olun.")
        mpu_initialized = False
    stdscr.refresh()
    return mpu_initialized

def read_raw_data(addr):
    if not mpu_initialized or not bus:
        return 0
    try:
        high = bus.read_byte_data(MPU6050_ADDR, addr)
        low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
        value = (high << 8) | low
        if value > 32767:
            value -= 65536
        return value
    except Exception as e:
        # print(f"MPU okuma hatası: {e}") # Debug için
        return 0

def get_gyro_x():
    gx_raw = read_raw_data(GYRO_XOUT_H)
    return gx_raw / 131.0

def calibrate_gyro_x(stdscr, duration=2.0):
    global offset_x
    if not mpu_initialized:
        stdscr.clear()
        stdscr.addstr(0,0, "MPU6050 başlatılmadığı için kalibrasyon yapılamıyor.")
        stdscr.refresh()
        time.sleep(2)
        offset_x = 0.0
        return offset_x

    stdscr.clear()
    stdscr.addstr(0, 0, "Kalibrasyon yapılıyor... Lütfen sensörü sabit tutun.")
    stdscr.refresh()

    samples = int(duration * 100)
    total_x_val = 0
    for i in range(samples):
        total_x_val += get_gyro_x()
        stdscr.addstr(1, 0, f"Kalibrasyon ilerlemesi: %{100 * (i+1) // samples}     ")
        stdscr.refresh()
        time.sleep(0.01)

    offset_x = total_x_val / samples
    stdscr.addstr(2, 0, f"Kalibrasyon tamamlandı. Ofset X: {offset_x:.2f} dps")
    stdscr.refresh()
    time.sleep(2)
    return offset_x

def setup_gpio_pins():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    for pin in direction_pins + pwm_hardware_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)

def initialize_pwms(stdscr):
    global pwm_m1, pwm_m2, pwm_m3, pwm_m4, pwm_led, pwm_initialized, led_pwm_initialized
    pwm_frequency = 100
    try:
        pwm_m1 = GPIO.PWM(PWM_M1_PIN, pwm_frequency)
        pwm_m2 = GPIO.PWM(PWM_M2_PIN, pwm_frequency)
        pwm_m3 = GPIO.PWM(PWM_M3_PIN, pwm_frequency)
        pwm_m4 = GPIO.PWM(PWM_M4_PIN, pwm_frequency)
        pwm_m1.start(0); pwm_m2.start(0); pwm_m3.start(0); pwm_m4.start(0)
        pwm_initialized = True
        if stdscr: stdscr.addstr(2,0, "Motor PWM'leri başarıyla başlatıldı.                     ")
    except Exception as e:
        if stdscr:
            stdscr.addstr(2,0, f"Motor PWM başlatılırken hata: {e}.                     ")
            stdscr.addstr(3,0, f"Kullanılan PWM pinleri: M1:{PWM_M1_PIN}, M2:{PWM_M2_PIN}, M3:{PWM_M3_PIN}, M4:{PWM_M4_PIN}")
            stdscr.addstr(4,0, "Lütfen bu GPIO pinlerinin doğru olduğundan emin olun.")
        pwm_initialized = False

    try:
        pwm_led = GPIO.PWM(LED_PIN, pwm_frequency)
        pwm_led.start(0)
        led_pwm_initialized = True
        if stdscr: stdscr.addstr(5,0, "LED PWM başarıyla başlatıldı.                          ")
    except Exception as e:
        if stdscr: stdscr.addstr(5,0, f"LED PWM başlatılırken hata: {e}                          ")
        led_pwm_initialized = False
    if stdscr: stdscr.refresh()
    return pwm_initialized and led_pwm_initialized

def set_motor_action(action, speed_percent):
    if not pwm_initialized:
        return

    calculated_speed = 0
    abs_speed = abs(speed_percent)

    if action == 'stop' or abs_speed < PWM_DEADBAND:
        calculated_speed = 0
    else:
        calculated_speed = max(MIN_SPEED, min(abs_speed, MAX_SPEED))

    if calculated_speed == 0:
        motor_durdur()
        return

    if action == 'forward' or action == 'ileri_a' or action == 'ileri_b': # ileri_a ve ileri_b için de aynı yön
        GPIO.output(M1_IN1, GPIO.LOW); GPIO.output(M1_IN2, GPIO.HIGH)
        GPIO.output(M2_IN3, GPIO.HIGH); GPIO.output(M2_IN4, GPIO.LOW)
        GPIO.output(M3_IN1, GPIO.LOW); GPIO.output(M3_IN2, GPIO.HIGH)
        GPIO.output(M4_IN3, GPIO.HIGH); GPIO.output(M4_IN4, GPIO.LOW)
    elif action == 'backward': # Şu an kullanılmıyor ama yapısı kalsın
        GPIO.output(M1_IN1, GPIO.HIGH); GPIO.output(M1_IN2, GPIO.LOW)
        GPIO.output(M2_IN3, GPIO.LOW); GPIO.output(M2_IN4, GPIO.HIGH)
        GPIO.output(M3_IN1, GPIO.HIGH); GPIO.output(M3_IN2, GPIO.LOW)
        GPIO.output(M4_IN3, GPIO.LOW); GPIO.output(M4_IN4, GPIO.HIGH)
    elif action == 'turn_left' or action == 'sola_don':
        GPIO.output(M1_IN1, GPIO.HIGH); GPIO.output(M1_IN2, GPIO.LOW)  # M1 Geri
        GPIO.output(M2_IN3, GPIO.LOW); GPIO.output(M2_IN4, GPIO.HIGH) # M2 Geri
        GPIO.output(M3_IN1, GPIO.LOW); GPIO.output(M3_IN2, GPIO.HIGH) # M3 İleri
        GPIO.output(M4_IN3, GPIO.HIGH); GPIO.output(M4_IN4, GPIO.LOW) # M4 İleri
    elif action == 'turn_right' or action == 'saga_don':
        GPIO.output(M1_IN1, GPIO.LOW); GPIO.output(M1_IN2, GPIO.HIGH) # M1 İleri
        GPIO.output(M2_IN3, GPIO.HIGH); GPIO.output(M2_IN4, GPIO.LOW) # M2 İleri
        GPIO.output(M3_IN1, GPIO.HIGH); GPIO.output(M3_IN2, GPIO.LOW) # M3 Geri
        GPIO.output(M4_IN3, GPIO.LOW); GPIO.output(M4_IN4, GPIO.HIGH) # M4 Geri

    if pwm_m1: pwm_m1.ChangeDutyCycle(calculated_speed)
    if pwm_m2: pwm_m2.ChangeDutyCycle(calculated_speed)
    if pwm_m3: pwm_m3.ChangeDutyCycle(calculated_speed)
    if pwm_m4: pwm_m4.ChangeDutyCycle(calculated_speed)

def motor_durdur():
    if not pwm_initialized: return
    if pwm_m1: pwm_m1.ChangeDutyCycle(0)
    if pwm_m2: pwm_m2.ChangeDutyCycle(0)
    if pwm_m3: pwm_m3.ChangeDutyCycle(0)
    if pwm_m4: pwm_m4.ChangeDutyCycle(0)
    for pin in direction_pins:
        GPIO.output(pin, GPIO.LOW)

class PIDController:
    def __init__(self, Kp, Ki, Kd, integral_limit=50.0, output_limit=(-100.0, 100.0)): # __init__ düzeltildi
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral_limit = integral_limit
        self.output_limit_min, self.output_limit_max = output_limit
        self.reset()

    def reset(self):
        self._integral = 0
        self._previous_error = 0

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        p_term = self.Kp * error

        if dt > 0:
            self._integral += error * dt
            self._integral = max(min(self._integral, self.integral_limit), -self.integral_limit)
        i_term = self.Ki * self._integral

        d_term = 0
        if dt > 0:
            derivative = (error - self._previous_error) / dt
            d_term = self.Kd * derivative

        output = p_term + i_term + d_term
        self._previous_error = error
        return max(min(output, self.output_limit_max), self.output_limit_min)

KP = 2.8
KI = 0.45
KD = 0.3
ANGLE_TOLERANCE = 0.7
TURN_TIMEOUT = 7.0

def turn_pid(stdscr, target_relative_angle, gyro_offset_param):
    for i in range(3, 10): # curses satırlarını temizle
        stdscr.move(i, 0)
        stdscr.clrtoeol()

    if not pwm_initialized or not mpu_initialized:
        message = "Hata: "
        if not pwm_initialized: message += "PWM başlatılmadı. "
        if not mpu_initialized: message += "MPU başlatılmadı. "
        message += "Dönüş yapılamaz."
        stdscr.addstr(3, 0, message.ljust(curses.COLS -1 if curses.COLS > 0 else 60))
        stdscr.refresh()
        time.sleep(2)
        return 0.0

    pid = PIDController(Kp=KP, Ki=KI, Kd=KD, integral_limit=70, output_limit=(-MAX_SPEED, MAX_SPEED))
    pid.reset()

    angle_turned_this_turn = 0.0
    start_time = time.time()
    prev_time = start_time
    stdscr.addstr(4, 0, f"Hedef Bağıl Açı: {target_relative_angle}°".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
    stdscr.addstr(5, 0, "PID Dönüş Başladı...".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
    stdscr.refresh()
    consecutive_low_error_count = 0
    min_consecutive_for_stop = 5

    while (time.time() - start_time) < TURN_TIMEOUT:
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        if dt <= 0:
            time.sleep(0.001)
            continue

        gyro_rate = get_gyro_x() - gyro_offset_param
        angle_increment = -gyro_rate * dt
        angle_turned_this_turn += angle_increment
        error = target_relative_angle - angle_turned_this_turn
        pid_output = pid.compute(target_relative_angle, angle_turned_this_turn, dt)

        current_turn_action = 'stop'
        speed_command_value_for_action = pid_output

        if abs(error) > ANGLE_TOLERANCE:
            if abs(pid_output) > PWM_DEADBAND:
                if pid_output > 0 :
                    current_turn_action = 'turn_left'
                elif pid_output < 0:
                    current_turn_action = 'turn_right'
        set_motor_action(current_turn_action, speed_command_value_for_action)

        stdscr.addstr(6, 0, f"Dönülen Açı: {angle_turned_this_turn:6.2f}° (Hedef: {target_relative_angle:+.1f}°)".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
        stdscr.addstr(7, 0, f"Hata: {error:6.2f}° | PID: {pid_output:6.2f} (Eylem: {current_turn_action})".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
        stdscr.addstr(8, 0, f"Jiro Hızı (işlenmiş): {-gyro_rate:.2f} dps | dt: {dt:.4f}s".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
        stdscr.refresh()

        if abs(error) < ANGLE_TOLERANCE:
            consecutive_low_error_count += 1
            if consecutive_low_error_count >= min_consecutive_for_stop and abs(pid_output) < EFFECTIVE_MIN_PWM :
                stdscr.addstr(9, 0, "Hedefe ulaşıldı, PID çıktısı düşük. Duruluyor...".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
                stdscr.refresh()
                break
        else:
            consecutive_low_error_count = 0

        if current_turn_action == 'stop' and abs(error) < ANGLE_TOLERANCE * 1.5:
            stdscr.addstr(9, 0, "PID dur komutu verdi, hata kabul edilebilir. Duruluyor...".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
            stdscr.refresh()
            break
    motor_durdur()
    final_message = f"Dönüş tamamlandı. Son Açı: {angle_turned_this_turn:.2f}° (Hata: {error:.2f}°)"
    stdscr.addstr(9, 0, final_message.ljust(curses.COLS-1 if curses.COLS > 0 else 60))
    stdscr.refresh()
    time.sleep(1.5)
    return angle_turned_this_turn

def connect_to_server(stdscr, host, port):
    global client_socket, connected_to_server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        stdscr.addstr(0, 0, f"{host}:{port} adresine bağlanılıyor...")
        stdscr.refresh()
        s.connect((host, port))
        client_socket = s
        connected_to_server = True
        stdscr.addstr(0, 0, f"{host}:{port} adresine başarıyla bağlandı! LED %50 PWM.")
        if led_pwm_initialized and pwm_led:
            pwm_led.ChangeDutyCycle(50)
        stdscr.refresh()
        return s
    except socket.timeout:
        stdscr.addstr(0, 0, f"Bağlantı zaman aşımına uğradı: {host}:{port}")
        stdscr.refresh()
        time.sleep(2)
        return None
    except Exception as e:
        stdscr.addstr(0, 0, f"Bağlantı hatası: {e}")
        stdscr.refresh()
        time.sleep(2)
        return None

def send_message(sock, message, stdscr):
    try:
        if sock:
            sock.sendall((message + "\n").encode('utf-8'))
            return True
    except Exception as e:
        if stdscr:
            stdscr.addstr(curses.LINES - 2, 0, f"Gönderme hatası: {e}".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
            stdscr.refresh()
        global connected_to_server
        connected_to_server = False
        if pwm_led: pwm_led.ChangeDutyCycle(0)
        return False
    return False

def receive_message(sock, stdscr):
    global connected_to_server
    try:
        if sock:
            sock.settimeout(1.0)
            data = sock.recv(4096)
            if not data:
                connected_to_server = False
                if pwm_led: pwm_led.ChangeDutyCycle(0)
                if stdscr:
                    stdscr.addstr(curses.LINES - 2, 0, "Sunucu bağlantıyı kapattı.".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
                    stdscr.refresh()
                return None
            return data.decode('utf-8').strip()
    except socket.timeout:
        return "TIMEOUT"
    except ConnectionResetError:
        connected_to_server = False
        if pwm_led: pwm_led.ChangeDutyCycle(0)
        if stdscr:
            stdscr.addstr(curses.LINES - 2, 0, "Bağlantı sıfırlandı.".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
            stdscr.refresh()
        return None
    except Exception as e:
        connected_to_server = False
        if pwm_led: pwm_led.ChangeDutyCycle(0)
        if stdscr:
            stdscr.addstr(curses.LINES - 2, 0, f"Alma hatası: {e}".ljust(curses.COLS-1 if curses.COLS > 0 else 60))
            stdscr.refresh()
        return None
    return None

def perform_stop_and_cleanup(stdscr_ref, main_socket, from_exception=False):
    global connected_to_server, client_socket, pwm_led, mpu_initialized, pwm_initialized, led_pwm_initialized
    global pwm_m1, pwm_m2, pwm_m3, pwm_m4

    if stdscr_ref:
        stdscr_ref.addstr(curses.LINES - 3, 0, "DURDUR komutu alındı/Hata oluştu. Temizleniyor...".ljust(curses.COLS-1 if curses.COLS >0 else 60))
        stdscr_ref.refresh()

    motor_durdur()

    if pwm_led:
        pwm_led.ChangeDutyCycle(0)
        pwm_led.stop()
        pwm_led = None
    led_pwm_initialized = False

    if pwm_initialized:
        if pwm_m1: pwm_m1.stop()
        if pwm_m2: pwm_m2.stop()
        if pwm_m3: pwm_m3.stop()
        if pwm_m4: pwm_m4.stop()
        pwm_m1, pwm_m2, pwm_m3, pwm_m4 = None, None, None, None
    pwm_initialized = False
    mpu_initialized = False

    if GPIO.getmode() is not None:
        GPIO.cleanup()
        if stdscr_ref:
            stdscr_ref.addstr(curses.LINES - 2, 0, "GPIO temizlendi.".ljust(curses.COLS-1 if curses.COLS >0 else 60))
            stdscr_ref.refresh()

    if main_socket:
        try:
            if not from_exception:
                send_message(main_socket, "STOP_ACK", stdscr_ref)
            main_socket.close()
        except Exception as e:
            if stdscr_ref:
                stdscr_ref.addstr(curses.LINES - 1, 0, f"Soket kapatma hatası (cleanup): {e}".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                stdscr_ref.refresh()

    client_socket = None
    connected_to_server = False
    if stdscr_ref and not from_exception:
        stdscr_ref.addstr(curses.LINES - 1, 0, "Temizlik tamamlandı. Program sonlandırıldı.".ljust(curses.COLS-1 if curses.COLS >0 else 60))
        stdscr_ref.refresh()
        time.sleep(2)

def led_celebrate_pattern(stdscr, sock_ref, duration_sec=10, interval=0.2):
    if not led_pwm_initialized or not pwm_led:
        if stdscr: stdscr.addstr(10, 0, "Kutlama LED'i başlatılamadı.".ljust(curses.COLS-1 if curses.COLS >0 else 60)); stdscr.refresh()
        return False

    if stdscr: stdscr.addstr(10, 0, "Kutlama LED'i aktif! (0-100 PWM döngüsü)".ljust(curses.COLS-1 if curses.COLS >0 else 60)); stdscr.refresh()

    start_time = time.time()
    on = True
    stop_received_during_celebration = False

    while time.time() - start_time < duration_sec:
        if sock_ref:
            message = receive_message(sock_ref, stdscr)
            if message == "STOP":
                if stdscr: stdscr.addstr(11, 0, "Kutlama sırasında STOP alındı!".ljust(curses.COLS-1 if curses.COLS >0 else 60)); stdscr.refresh()
                stop_received_during_celebration = True
                break
            elif message is None and not connected_to_server:
                if stdscr: stdscr.addstr(11, 0, "Kutlama sırasında bağlantı kesildi.".ljust(curses.COLS-1 if curses.COLS >0 else 60)); stdscr.refresh()
                stop_received_during_celebration = True
                break

        pwm_led.ChangeDutyCycle(100 if on else 0)
        on = not on
        time.sleep(interval)

    pwm_led.ChangeDutyCycle(0)
    if stdscr and not stop_received_during_celebration:
        stdscr.addstr(10, 0, "Kutlama LED'i tamamlandı.                                         ".ljust(curses.COLS-1 if curses.COLS >0 else 60))
        stdscr.refresh()
    return stop_received_during_celebration

def parse_commands(command_str, stdscr):
    try:
        parsed_sequence = json.loads(command_str)
        if not isinstance(parsed_sequence, list):
            stdscr.addstr(12, 0, "Hata: Komutlar liste formatında değil.".ljust(curses.COLS-1 if curses.COLS >0 else 60))
            stdscr.refresh()
            return None
        valid_commands = []
        for cmd_tuple in parsed_sequence:
            if isinstance(cmd_tuple, list) and len(cmd_tuple) == 2:
                action, value = cmd_tuple
            elif isinstance(cmd_tuple, tuple) and len(cmd_tuple) == 2:
                action, value = cmd_tuple
            else:
                stdscr.addstr(12, 0, f"Hata: Geçersiz komut formatı: {cmd_tuple}".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                stdscr.refresh()
                return None
            
            # ileri_a ve ileri_b komutlarını da kabul et
            if not isinstance(action, str) or not isinstance(value, int) or \
               not (action in ["ileri_a", "ileri_b", "sola_don", "saga_don"]): # "ileri" kaldırıldı, yerine a/b geldi
                stdscr.addstr(12, 0, f"Hata: Geçersiz komut tipi/eylemi: {action}, {value}".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                stdscr.refresh()
                return None
            valid_commands.append((action, value))
        return valid_commands
    except json.JSONDecodeError:
        stdscr.addstr(12, 0, "Hata: Komutlar JSON formatında değil.".ljust(curses.COLS-1 if curses.COLS >0 else 60))
        stdscr.refresh()
        return None
    except Exception as e:
        stdscr.addstr(12, 0, f"Komut ayrıştırma hatası: {e}".ljust(curses.COLS-1 if curses.COLS >0 else 60))
        stdscr.refresh()
        return None

def main_loop(stdscr):
    global client_socket, connected_to_server, mpu_initialized, pwm_initialized, led_pwm_initialized
    global offset_x, current_image_source_on_pi
    global current_adim_kazanci_a, current_adim_kazanci_b # Aktif kazançlar

    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.clear()

    setup_gpio_pins()
    if not initialize_pwms(stdscr):
        stdscr.addstr(6,0, "PWM başlatılamadı. Program sonlandırılıyor.")
        stdscr.refresh()
        time.sleep(3)
        return

    while True:
        if not connected_to_server:
            stdscr.clear()
            stdscr.addstr(0,0, "Sunucuya bağlanmaya çalışılıyor...")
            stdscr.refresh()
            client_socket = connect_to_server(stdscr, SERVER_HOST, SERVER_PORT)
            if not client_socket:
                stdscr.addstr(1,0, f"{SERVER_HOST}:{SERVER_PORT} adresine bağlanılamadı. 5 saniye sonra tekrar denenecek.")
                stdscr.refresh()
                time.sleep(5)
                continue

        stdscr.clear()
        stdscr.addstr(0, 0, "Sunucuya bağlı. Komut bekleniyor (CALIBRATE:[KAYNAK], COMMANDS:{...}, STOP)".ljust(curses.COLS-1 if curses.COLS >0 else 80))
        stdscr.refresh()

        message = receive_message(client_socket, stdscr)

        if message is None:
            if connected_to_server:
                connected_to_server = False
                if pwm_led: pwm_led.ChangeDutyCycle(0)
            stdscr.addstr(1, 0, "Sunucu bağlantısı kesildi. Yeniden bağlanılacak...".ljust(curses.COLS-1 if curses.COLS >0 else 80))
            stdscr.refresh()
            if client_socket: client_socket.close(); client_socket = None
            time.sleep(1)
            continue

        if message == "TIMEOUT":
            continue

        stdscr.addstr(1,0, f"Alınan Mesaj: {message[:60]}".ljust(curses.COLS-1 if curses.COLS >0 else 80))
        stdscr.refresh()

        if message.startswith("CALIBRATE:"):
            parts = message.split(":")
            if len(parts) == 2:
                current_image_source_on_pi = parts[1].upper()
                if current_image_source_on_pi == "GALLERY":
                    current_adim_kazanci_a = ADIM_KAZANCI_GALLERY_A
                    current_adim_kazanci_b = ADIM_KAZANCI_GALLERY_B
                else: # Varsayılan olarak veya "CAMERA" için
                    current_adim_kazanci_a = ADIM_KAZANCI_CAMERA_A
                    current_adim_kazanci_b = ADIM_KAZANCI_CAMERA_B
                stdscr.addstr(2,0, f"Kaynak: {current_image_source_on_pi}, Kznç A: {current_adim_kazanci_a:.3f}, Kznç B: {current_adim_kazanci_b:.3f}".ljust(curses.COLS-1 if curses.COLS >0 else 80))
            else:
                current_image_source_on_pi = "UNKNOWN"
                current_adim_kazanci_a = ADIM_KAZANCI_CAMERA_A # Bilinmiyorsa varsayılan
                current_adim_kazanci_b = ADIM_KAZANCI_CAMERA_B
                stdscr.addstr(2,0, f"Kalibrasyon: Kaynak yok. Varsayılan kazançlar A:{current_adim_kazanci_a:.3f} B:{current_adim_kazanci_b:.3f}".ljust(curses.COLS-1 if curses.COLS >0 else 80))
            stdscr.refresh()
            
            if not mpu_initialized:
                initialize_mpu(stdscr)
            
            current_offset = calibrate_gyro_x(stdscr)
            if mpu_initialized:
                if pwm_led: pwm_led.ChangeDutyCycle(0)
                send_message(client_socket, f"CALIBRATION_DONE:{current_offset}", stdscr)
            else:
                send_message(client_socket, "CALIBRATION_FAIL:MPU_INIT_ERROR", stdscr)
                if pwm_led: pwm_led.ChangeDutyCycle(0)

        elif message.startswith("COMMANDS:"):
            command_data_str = message[len("COMMANDS:"):]
            COMMAND_SEQUENCE = parse_commands(command_data_str, stdscr)

            if COMMAND_SEQUENCE:
                send_message(client_socket, "COMMANDS_RECEIVED_VALID", stdscr)
                stdscr.clear()
                stdscr.addstr(0,0, "Komutlar alındı ve yürütülüyor...")
                stdscr.refresh()
                
                current_total_angle_estimate = 0.0
                
                if not mpu_initialized:
                    stdscr.addstr(1,0,"UYARI: MPU başlatılmadı, dönüşler hatalı olabilir/atlanabilir!")
                    stdscr.refresh()
                    time.sleep(1)

                gyro_offset_for_turns = offset_x

                for cmd_idx, (action, value) in enumerate(COMMAND_SEQUENCE):
                    stdscr.move(1,0); stdscr.clrtoeol()
                    stdscr.addstr(1, 0, f"Komut Dizisi: {cmd_idx + 1}/{len(COMMAND_SEQUENCE)}".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                    stdscr.move(2,0); stdscr.clrtoeol()
                    stdscr.addstr(2, 0, f"Toplam Tahmini Yön: {current_total_angle_estimate:.2f}° (Ofset: {gyro_offset_for_turns:.2f})".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                    
                    current_action_message = f"İşleniyor: {action}"
                    effective_steps = 0
                    selected_kazanc = 0

                    if action == "ileri_a":
                        selected_kazanc = current_adim_kazanci_a
                        effective_steps = value * selected_kazanc
                        current_action_message += f" (Yatay {value}x{selected_kazanc:.3f}={effective_steps:.0f} adım)"
                    elif action == "ileri_b":
                        selected_kazanc = current_adim_kazanci_b
                        effective_steps = value * selected_kazanc
                        current_action_message += f" (Dikey {value}x{selected_kazanc:.3f}={effective_steps:.0f} adım)"
                    elif "don" in action:
                        current_action_message += " (90 derece)"
                    
                    stdscr.move(3,0); stdscr.clrtoeol()
                    stdscr.addstr(3, 0, current_action_message.ljust(curses.COLS-1 if curses.COLS >0 else 60))
                    stdscr.refresh()

                    if action == "ileri_a" or action == "ileri_b":
                        duration = FORWARD_DURATION_PER_STEP * effective_steps
                        status_msg = f"İleri hareket ({duration:.1f}s, Hız: {FORWARD_SPEED}%)..."
                        stdscr.move(4,0); stdscr.clrtoeol()
                        stdscr.addstr(4, 0, status_msg.ljust(curses.COLS-1 if curses.COLS >0 else 60)); stdscr.refresh()
                        set_motor_action(action, FORWARD_SPEED) # action 'ileri_a' veya 'ileri_b' olabilir, set_motor_action bunu 'forward' gibi ele alır
                        time.sleep(duration)
                        motor_durdur()
                        stdscr.move(4,0); stdscr.clrtoeol()
                        stdscr.addstr(4, 0, f"İleri hareket tamamlandı ({duration:.1f}s).".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                        stdscr.refresh()

                    elif action == "sola_don":
                        if mpu_initialized:
                            angle_this_turn = turn_pid(stdscr, 90.0, gyro_offset_for_turns)
                            current_total_angle_estimate += angle_this_turn
                        else:
                            stdscr.addstr(4, 0, "MPU yok, sola dönüş atlandı.".ljust(curses.COLS-1 if curses.COLS >0 else 60)); stdscr.refresh(); time.sleep(1)
                        motor_durdur()

                    elif action == "saga_don":
                        if mpu_initialized:
                            angle_this_turn = turn_pid(stdscr, -90.0, gyro_offset_for_turns)
                            current_total_angle_estimate += angle_this_turn
                        else:
                            stdscr.addstr(4, 0, "MPU yok, sağa dönüş atlandı.".ljust(curses.COLS-1 if curses.COLS >0 else 60)); stdscr.refresh(); time.sleep(1)
                        motor_durdur()
                    else:
                        stdscr.move(4,0); stdscr.clrtoeol()
                        stdscr.addstr(4, 0, f"Bilinmeyen komut: {action}".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                        stdscr.refresh()

                    stop_check_during_delay = receive_message(client_socket, stdscr)
                    if stop_check_during_delay == "STOP":
                        perform_stop_and_cleanup(stdscr, client_socket)
                        return
                    elif stop_check_during_delay is None and not connected_to_server:
                        if client_socket: client_socket.close(); client_socket = None
                        break 

                    if cmd_idx < len(COMMAND_SEQUENCE) - 1:
                        stdscr.move(10,0); stdscr.clrtoeol()
                        stdscr.addstr(10, 0, f"{DELAY_BETWEEN_COMMANDS} saniye bekleniyor...".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                        stdscr.refresh()
                        time.sleep(DELAY_BETWEEN_COMMANDS)
                        stdscr.move(10,0); stdscr.clrtoeol()
                
                if connected_to_server: # Eğer komut döngüsü bağlantı kopmasıyla kesilmediyse
                    stdscr.move(11,0); stdscr.clrtoeol()
                    stdscr.addstr(11, 0, "Tüm komutlar tamamlandı.".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                    send_message(client_socket, "SEQUENCE_DONE", stdscr)
                    stdscr.refresh()
                    
                    stop_after_celeb = led_celebrate_pattern(stdscr, client_socket)
                    if stop_after_celeb:
                        perform_stop_and_cleanup(stdscr, client_socket)
                        return
                    elif not connected_to_server: # Kutlama sırasında bağlantı koptuysa
                        if client_socket: client_socket.close(); client_socket = None
                        continue
            else:
                send_message(client_socket, "COMMANDS_INVALID_FORMAT", stdscr)
                stdscr.addstr(12, 0, "Geçersiz komut formatı, sunucuya bildirildi.".ljust(curses.COLS-1 if curses.COLS >0 else 60))
                stdscr.refresh()
                time.sleep(2)

        elif message == "STOP":
            stdscr.addstr(curses.LINES-3, 0, "STOP komutu alındı. Durduruluyor...".ljust(curses.COLS-1 if curses.COLS >0 else 60))
            stdscr.refresh()
            perform_stop_and_cleanup(stdscr, client_socket)
            return

        elif message:
            stdscr.addstr(curses.LINES-3,0, f"Bilinmeyen mesaj: {message[:50]}".ljust(curses.COLS-1 if curses.COLS >0 else 60))
            stdscr.refresh()
            time.sleep(1)

def curses_main_wrapper(stdscr):
    try:
        main_loop(stdscr)
    except KeyboardInterrupt:
        if stdscr:
            stdscr.addstr(curses.LINES - 1, 0, "Ctrl+C algılandı. Temizleniyor...".ljust(curses.COLS-1 if curses.COLS >0 else 60))
            stdscr.refresh()
            time.sleep(1)
        perform_stop_and_cleanup(stdscr, client_socket, from_exception=True)
    except Exception as e:
        if stdscr:
            stdscr.clear()
            stdscr.addstr(0,0, "Kritik bir hata oluştu! Detaylar konsolda olabilir.")
            stdscr.addstr(1,0, f"Hata: {str(e)[:curses.COLS-5 if curses.COLS > 5 else curses.COLS-1]}")
            stdscr.refresh()
            time.sleep(3)
        perform_stop_and_cleanup(stdscr, client_socket, from_exception=True)
        raise

if __name__ == '__main__':
    try:
        offset_x = 0.0
        # Başlangıç kazançlarını ayarla (CALIBRATE ile değişebilir)
        current_image_source_on_pi = "CAMERA" # Varsayılan
        current_adim_kazanci_a = ADIM_KAZANCI_CAMERA_A
        current_adim_kazanci_b = ADIM_KAZANCI_CAMERA_B
        curses.wrapper(curses_main_wrapper)
    except Exception as e_outer:
        print(f"Program başlatılırken veya çalışırken genel bir hata oluştu: {e_outer}")
        traceback.print_exc()
    finally:
        if GPIO.getmode() is not None:
            print("Ana program sonu: Ek GPIO temizliği yapılıyor (gerekirse).")
            try:
                if 'pwm_m1' in globals() and pwm_m1: pwm_m1.stop()
                if 'pwm_m2' in globals() and pwm_m2: pwm_m2.stop()
                if 'pwm_m3' in globals() and pwm_m3: pwm_m3.stop()
                if 'pwm_m4' in globals() and pwm_m4: pwm_m4.stop()
                if 'pwm_led' in globals() and pwm_led: pwm_led.stop()
            except RuntimeError:
                pass
            except NameError:
                pass
            GPIO.cleanup()
            print("GPIO temizlendi (ana program sonu).")
        print("Program sonlandırıldı.")
