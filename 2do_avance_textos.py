#INTERFAZ CON SELECCION DE DISPOSITIVOS CONECTADOS
import flet as ft
import cv2
import base64
import threading
import time
import depthai as dai
import numpy as np  # <-- añadido para el filtro Sharpened

def main(page: ft.Page):
    cap = None
    running = False
    filtro_actual = "original"
    thread = None
    first_frame_rendered = False
    # cam_mode = None  # por defecto

    # Imagen de la cámara
    img = ft.Image(width=640, height=480, visible=False)

    # Overlay de carga
    loading_overlay = ft.Container(
        content=ft.Column(
            [
                ft.ProgressRing(width=60, height=60, stroke_width=6, color="#18A558"),
                ft.Text("Iniciando cámara...", color=ft.Colors.BLACK)
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        ),
        alignment=ft.alignment.center,
        bgcolor=ft.Colors.WHITE,
        visible=False,
        expand=True,
    )

    # ==== Filtros ====
    def aplicar_filtro(frame):
        nonlocal filtro_actual
        if filtro_actual == "gris":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif filtro_actual == "canny":
            return cv2.Canny(frame, 100, 200)
        elif filtro_actual == "blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif filtro_actual == "invert":
            return cv2.bitwise_not(frame)
        elif filtro_actual == "sharpened":
            # Kernel de enfoque (sharpen)
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
            return cv2.filter2D(frame, -1, kernel)
        return frame

    # ==== Captura Webcam ====
    def capture_webcam():
        nonlocal running, cap, first_frame_rendered
        while running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = aplicar_filtro(frame)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            _, buffer = cv2.imencode(".jpg", frame)
            img.src_base64 = base64.b64encode(buffer).decode("utf-8")
            if not first_frame_rendered:
                first_frame_rendered = True
                img.visible = True
                loading_overlay.visible = False
            page.update()
            time.sleep(0.03)
        if cap is not None:
            cap.release()

    # ==== Captura OAK ====
    def capture_oak():
        nonlocal running, first_frame_rendered
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        xout = pipeline.createXLinkOut()
        xout.setStreamName("video")
        cam_rgb.preview.link(xout.input)

        with dai.Device(pipeline) as device:
            q = device.getOutputQueue("video", maxSize=4, blocking=False)
            while running:
                in_frame = q.get()
                frame = in_frame.getCvFrame()
                frame = aplicar_filtro(frame)
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                _, buffer = cv2.imencode(".jpg", frame)
                img.src_base64 = base64.b64encode(buffer).decode("utf-8")
                if not first_frame_rendered:
                    first_frame_rendered = True
                    img.visible = True
                    loading_overlay.visible = False
                page.update()
                time.sleep(0.03)

    # ==== Activar cámara ====
    def act_camara(e):
        nonlocal cap, running, thread, first_frame_rendered, cam_mode
        try:
            running = True
            first_frame_rendered = False
            loading_overlay.visible = True
            img.visible = False
            page.update()

            # if cam_mode == "webcam":
            #     cap = cv2.VideoCapture(0)  # Cambia a 1,2 si tienes varias
            #     if not cap.isOpened():
            #         raise Exception("No se pudo acceder a la webcam")
            #     thread = threading.Thread(target=capture_webcam, daemon=True)
            # else:
            #     thread = threading.Thread(target=capture_oak, daemon=True)
            if cam_mode.startswith("Webcam"):
                index = int(cam_mode.split(" ")[1])
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    raise Exception(f"No se pudo acceder a la {cam_mode}")
                thread = threading.Thread(target=capture_webcam, daemon=True)

            elif cam_mode == "OAK":
                thread = threading.Thread(target=capture_oak, daemon=True)

            thread.start()
        except Exception as ex:
            loading_overlay.visible = False
            page.update()
            dlg = ft.AlertDialog(title=ft.Text("Error"), content=ft.Text(str(ex)))
            page.open(dlg)

    # ==== Desactivar cámara ====
    def desact_camara(e):
        nonlocal running, thread, cap
        running = False
        if thread is not None:
            thread.join(timeout=1.0)
        if cap is not None:
            cap.release()
        img.visible = False
        loading_overlay.visible = False
        page.update()

    # ==== Salir ====
    def stop_cam(e):
        nonlocal running, thread
        running = False
        if thread is not None:
            thread.join(timeout=1.0)
        page.window.close()

    # ===== Helpers para UI de filtros (solo color activo) =====
    DEFAULT_GREEN = "#90EE90"
    ACTIVE_GREEN = "#2DBE60"

    # referencias a botones para poder marcar activo
    btn_original = None
    btn_gris = None
    btn_canny = None
    btn_blur = None
    btn_invert = None
    btn_sharp = None

    def marcar_activo(btn_activo):
        # Pone todos en verde claro y el activo en verde subido
        for b in [btn_original, btn_gris, btn_canny, btn_blur, btn_invert, btn_sharp]:
            if b is not None:
                b.bgcolor = DEFAULT_GREEN
        if btn_activo is not None:
            btn_activo.bgcolor = ACTIVE_GREEN

    # --- Descripción de filtros (Markdown) ---
    # Panel solicitado, estable y vacío por defecto
    desc_text = ft.Markdown(  # <-- ahora Markdown
        value="",
        selectable=True,
        code_theme="github",
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB
    )
    desc_panel = ft.Container(
        content=ft.Container(
            content=ft.Column([desc_text],scroll="auto"),
            expand=True,
            alignment=ft.alignment.top_left,
            padding=4
        ),
        width=640,
        padding=12,
        bgcolor=ft.Colors.WHITE,
        border_radius=16,
        border=ft.border.all(1, ft.Colors.BLACK12),
        shadow=[ft.BoxShadow(blur_radius=18, spread_radius=1, color=ft.Colors.BLACK26, offset=ft.Offset(0, 6))],
        expand=True,
    )

    # ==== Filtros ====
    def FiltroNormal(e):  
        nonlocal filtro_actual; filtro_actual = "original"
        marcar_activo(btn_original)
        desc_text.value = """# Original
**Descripción:** Imagen sin procesamiento adicional.

"""
        page.update()

    def Filtro1(e):       
        nonlocal filtro_actual; filtro_actual = "gris"
        marcar_activo(btn_gris)
        desc_text.value = """# Escala de grises
**Descripción:** Es una transformación que convierte una imagen de color (3 canales BGR) a una imagen de un solo canal en escala de grises, donde cada píxel representa solo la intensidad de la luz.
Utiliza una fórmula ponderada que imita la percepción del ojo humano, que es más sensible al verde y menos al azul:
```python
Gris = (0.299 * Valor_Rojo) + (0.587 * Valor_Verde) + (0.114 * Valor_Azul)
```

```python
gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```"""
        page.update()

    def Filtro2(e):       
        nonlocal filtro_actual; filtro_actual = "canny"
        marcar_activo(btn_canny)
        desc_text.value = """# Canny (detección de bordes)
**Descripción:** Combina la detección de bordes basada en gradientes con lógica avanzada para garantizar que los bordes detectados sean finos, estén conectados y sin ruido.
El filtro Canny es un algoritmo de detección de bordes desarrollado por John F. Canny en 1986. Es ampliamente utilizado en procesamiento de imágenes y visión por computadora.
Esto se logra aplicando un filtro Gaussiano para reducir el ruido, con kernel definido por:
```python
G(x, y) = (1 / (2 * π * σ^2)) * exp(-((x^2 + y^2) / (2 * σ^2)))
```
Ademas se calcula magnitudes y direcciones del gradiente usando operadores Sobel:
```python
Gx = ∂I/∂x, Gy = ∂I/∂y
Magnitud = √(Gx² + Gy²)
Dirección = atan2(Gy, Gx)
```
Manteniendo solo los píxeles que son máximos locales en la dirección del gradiente.

```python
edges = cv2.Canny(frame, 100, 200)
```"""
        page.update()

    def Filtro3(e):       
        nonlocal filtro_actual; filtro_actual = "blur"
        marcar_activo(btn_blur)
        desc_text.value = """# Desenfoque Gaussiano
**Descripción:** El desenfoque, también conocido como suavizado, es una operación fundamental en el procesamiento de imágenes que busca reducir el detalle y el ruido en una imagen.
Se utiliza a menudo como paso de preprocesamiento en tareas como la detección de bordes, el reconocimiento de objetos o incluso en efectos artísticos como suavizar una foto.
Explicado matematicamente hablando, quedaria de la siguiente manera:
El kernel Gaussiano se define como:
```python
G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
```
Propiedades:

-Suavizado isotrópico (igual en todas direcciones)

-Preserva mejor los bordes que otros filtros de suavizado

-Elimina ruido de alta frecuencia

```python
blur = cv2.GaussianBlur(frame, (15, 15), 0)
```"""
        page.update()

    def Filtro4(e):       
        nonlocal filtro_actual; filtro_actual = "invert"
        marcar_activo(btn_invert)
        desc_text.value = """# Invertir colores
**Descripción:** Una transformación que crea el negativo de la imagen, invirtiendo todos los valores de color.

¿Cómo funciona?
Opera a nivel de bits sobre cada canal de color (B, G, R) de cada píxel usando la operación NOT. Matemáticamente, para un valor de píxel X (entre 0 y 255), el nuevo valor es 255 - X.

El blanco (255) se convierte en negro (0).

El negro (0) se convierte en blanco (255).

Los colores intermedios se transforman en su complementario.
Para cada píxel en cada canal:
```python
I_out(x,y,c) = 255 - I_in(x,y,c)
```
Fundamento en álgebra booleana:
La operación es equivalente a NOT bit a bit:
```python
NOT(11110000) = 00001111
```

```python
invertido = cv2.bitwise_not(frame)
```"""
        page.update()

    def Filtro5(e):       
        nonlocal filtro_actual; filtro_actual = "sharpened"
        marcar_activo(btn_sharp)
        desc_text.value = """# Sharpened (enfoque)
**Descripción:** El enfoque realiza los bordes y los detalles finos de una imagen, haciéndola parecer más nítida y definida.

Para lograr esto, utilizamos un núcleo de convolución que hace dos cosas simultáneamente:

1.-Conserva el píxel central con un peso positivo.

2.-Resta la influencia de los píxeles circundantes utilizando pesos negativos.

El kernel clásico de enfoque que describes es:

[ 0, -1,  0]

[-1,  5, -1]

[ 0, -1,  0]

¿Por qué estos números? La lógica es brillante:

El centro (5): Tiene un peso positivo y grande. Su trabajo es amplificar el píxel actual.

Los vecinos (-1): Tienen pesos negativos. Su trabajo es restar o suprimir la influencia de los píxeles que lo rodean.

El efecto combinado es:
```python
Nuevo_Valor = (5 * Píxel_Central) - (Píxel_Arriba) - (Píxel_Abajo) - (Píxel_Izquierda) - (Píxel_Derecha)
```

```python
import numpy as np
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)
sharpened = cv2.filter2D(frame, -1, kernel)
```"""
        page.update()

    # ==== Cambio de modo cámara ====
    def cambiar_modo(e):
        nonlocal cam_mode
        cam_mode = e.control.value

    # ==== Interfaz ====
    page.title = "Webcam / OAK con Filtros - Flet OpenCV"
    page.window.width = 900
    page.window.height = 700
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = ft.Colors.WHITE

    btn_style = dict(bgcolor=DEFAULT_GREEN, color=ft.Colors.BLACK)
    cam_btn_style = dict(bgcolor="#333333", color=ft.Colors.WHITE)
    panel_bg = ft.Colors.WHITE
    panel_radius = 16
    panel_border = ft.border.all(1, ft.Colors.BLACK12)
    panel_shadow = [
        ft.BoxShadow(
            blur_radius=18, spread_radius=1,
            color=ft.Colors.BLACK26, offset=ft.Offset(0, 6)
        )
    ]

    BTN_WIDTH_SMALL = 130
    BTN_HEIGHT = 40

    #==Vista de Dispositivos Conectados==
    def listar_dispositivos(max_index=1):
        dispositivos = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                dispositivos.append(f"Webcam {i}")
                cap.release()
        try:
            devices = dai.Device.getAllAvailableDevices()
            if len(devices) > 0:
                dispositivos.append("OAK")
        except:
            pass
        return dispositivos

    dispositivos = listar_dispositivos()
    cam_mode = dispositivos[0] if dispositivos else None   # ✅ sincroniza valor inicial

    # Dropdown dinámico
    mode_selector = ft.Dropdown(
        options=[ft.dropdown.Option(d) for d in dispositivos],
        value=cam_mode,
        on_change=cambiar_modo,
        width=200
    )

    # Top bar
    top_bar = ft.Container(
        content=ft.Row(
            [
                ft.ElevatedButton("Activar Cámara", on_click=act_camara,
                                  icon=ft.Icons.VIDEOCAM, **cam_btn_style),
                ft.ElevatedButton("Desactivar Cámara", on_click=desact_camara,
                                  icon=ft.Icons.VIDEOCAM_OFF, **cam_btn_style),
                mode_selector,
                ft.Container(expand=1),
                ft.ElevatedButton("Salir", on_click=stop_cam,
                                  icon=ft.Icons.EXIT_TO_APP,
                                  bgcolor="#F28B82", color=ft.Colors.BLACK, height=BTN_HEIGHT)
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        ),
        padding=12,
        bgcolor=panel_bg,
        border_radius=panel_radius,
        border=panel_border,
        shadow=panel_shadow,
        margin=ft.margin.only(left=12, right=12, top=12, bottom=8),
    )

    # Panel de botones de filtros (referencias de botones)
    btn_original = ft.ElevatedButton("Original", on_click=FiltroNormal, icon=ft.Icons.CAMERA_ALT,
                                     width=BTN_WIDTH_SMALL, height=BTN_HEIGHT, **btn_style)
    btn_gris = ft.ElevatedButton("Gris", on_click=Filtro1, icon=ft.Icons.GRAIN,
                                 width=BTN_WIDTH_SMALL, height=BTN_HEIGHT, **btn_style)
    btn_canny = ft.ElevatedButton("Canny", on_click=Filtro2, icon=ft.Icons.BORDER_STYLE,
                                  width=BTN_WIDTH_SMALL, height=BTN_HEIGHT, **btn_style)
    btn_blur = ft.ElevatedButton("Blur", on_click=Filtro3, icon=ft.Icons.BLUR_ON,
                                 width=BTN_WIDTH_SMALL, height=BTN_HEIGHT, **btn_style)
    btn_invert = ft.ElevatedButton("Invertir", on_click=Filtro4, icon=ft.Icons.INVERT_COLORS,
                                   width=BTN_WIDTH_SMALL, height=BTN_HEIGHT, **btn_style)
    btn_sharp = ft.ElevatedButton("Sharpened", on_click=Filtro5, icon=ft.Icons.AUTO_FIX_HIGH,
                                  width=BTN_WIDTH_SMALL, height=BTN_HEIGHT, **btn_style)

    filtros_row = ft.Row(
        controls=[btn_original, btn_gris, btn_canny, btn_blur, btn_invert, btn_sharp],
        spacing=10, run_spacing=10,
        alignment=ft.MainAxisAlignment.START,
        wrap=True,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )

    # Panel de filtros (botones + panel de descripción ESTABLE debajo)
    filtros_tab_content = ft.Container(
        content=ft.Column(
            [filtros_row, ft.Container(height=8), desc_panel],
            spacing=8,
            alignment=ft.MainAxisAlignment.START,
            expand=True,
        ),
        width=640,
        padding=12,
        bgcolor=panel_bg,
        border_radius=panel_radius,
        border=panel_border,
        shadow=panel_shadow,
        expand=True,
    )

    tabs = ft.Tabs(
        selected_index=0,
        indicator_color="#90EE90",
        divider_color=ft.Colors.BLACK12,
        tabs=[
            ft.Tab(
                text="Filtros",
                content=filtros_tab_content,
                tab_content=ft.Text("Filtros", color=ft.Colors.BLACK)
            )
        ],
        expand=1
    )

    right_panel = ft.Container(
        content=ft.Column([tabs], alignment=ft.MainAxisAlignment.START, expand=True),
        padding=16, bgcolor=panel_bg,
        border_radius=panel_radius, border=panel_border,
        shadow=panel_shadow, margin=ft.margin.only(left=12, right=12),
        expand=True, clip_behavior=ft.ClipBehavior.NONE
    )

    camera_inner = ft.Stack(
        [
            ft.Container(bgcolor=ft.Colors.WHITE, expand=True),
            ft.Container(content=img, padding=8, alignment=ft.alignment.center, expand=True),
            loading_overlay
        ],
        expand=True,
    )

    camera_panel = ft.Container(
        content=camera_inner, bgcolor=panel_bg,
        border_radius=panel_radius, border=panel_border,
        shadow=panel_shadow, padding=8,
        margin=ft.margin.only(left=4, right=4),
        alignment=ft.alignment.center, expand=True,
    )

    main_row = ft.ResponsiveRow(
        controls=[
            ft.Container(content=camera_panel, col={"xs": 12, "md": 7}, expand=True),
            ft.Container(content=right_panel, col={"xs": 12, "md": 5}, expand=True),
        ],
        expand=True, vertical_alignment=ft.CrossAxisAlignment.START,
        alignment=ft.MainAxisAlignment.START,
    )

    layout = ft.Column([top_bar, main_row], expand=True)
    page.add(layout)

    # Marcar "Original" como activo por defecto en UI
    marcar_activo(btn_original)

    def on_window_event(e: ft.WindowEvent):
        if e.data == "close":
            stop_cam(e)
    page.window.on_event = on_window_event

ft.app(target=main,view=ft.AppView.WEB_BROWSER)
