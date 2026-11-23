# M0 · Windows desktop packaging spike

เลือก **pywebview + Microsoft Edge WebView2** สำหรับ Windows x64 และแพ็ก shell แบบ PyInstaller onedir โปรแกรมเปิดจาก executable แล้วสร้าง Python worker ที่ฟังบน loopback port ที่ OS จัดให้ หน้า React และข้อมูล workspace มาจาก origin เดียวกัน ทั้งสอง process สื่อสารเรื่อง lifecycle ผ่าน pipe

เหตุผลที่เลือกคือ renderer ใช้ JavaScript ที่ build แล้วได้โดยตรง ผู้ใช้เปิดโปรแกรมจากไดเรกทอรีที่คัดลอกไปใช้ได้ และเราทดสอบการแยก Python runtime ได้จริง เลือกแจก inference runtime แยกตาม profile CPU หรือ CUDA ก่อน M2 เพื่อให้เครื่อง CPU ติดตั้งเฉพาะ dependencies ที่ต้องใช้ การตัดสินใจเรื่องวิธีติดตั้ง PyTorch ขั้นสุดท้ายขึ้นกับขนาดและผลตรวจบน Windows สะอาดใน M3/M4

## หลักฐานที่ทดสอบบนเครื่องพัฒนา

| รายการ | สิ่งที่ตรวจและผล |
|---|---|
| Build | Windows 11 x64 build 26200, Python 3.12.5, PyInstaller 6.22.3, pywebview 6.2.1, React 19.3.1, esbuild 0.28.2, WebView2 Runtime ที่ติดตั้งบนเครื่อง 140.0.3485.66 |
| ขนาด | 34,269,391 bytes สำหรับ onedir shell รวม bundled Python, native folder picker และ React assets บนดิสก์ ขนาดนี้ไม่รวม WebView2 Runtime ของระบบและ PyTorch profile |
| ติดตั้งแยกจาก checkout | Smoke test คัดลอกทั้ง onedir package ไปยัง temporary directory เปิดจาก workspace ชื่อภาษาไทยที่มีช่องว่าง และตั้ง PATH ของโปรแกรมให้เหลือเพียง Windows กับ System32 |
| โปรแกรมที่ผู้ใช้เปิด | ทดสอบสองเส้นทางจาก packaged executable เดียวกัน คือ worker ที่มากับ shell และ worker ที่เปิดด้วย executable ของ project venv ที่ระบุด้วย absolute path |
| การแสดงผล | ตรวจข้อความชื่อโฟลเดอร์จริงจาก DOM หลัง React fetch ข้อมูลจาก Python worker ผ่าน WebView2 ที่ซ่อนหน้าต่างไว้ |
| เวลาจนเห็นผล | จากก่อน spawn worker ถึงอ่านชื่อ workspace ได้ สี่รอบต่อเนื่อง bundled 3.295 / 1.754 / 2.485 / 2.517 วินาที, external 2.228 / 2.084 / 2.134 / 1.853 วินาที |
| ปิด | แปดรันข้างต้นปิดหน้าต่างแล้ว launcher ปิด pipe ให้ worker shutdown และตรวจ exit code 0 ภายใน 5 วินาที |

หลังเพิ่ม timeout สำหรับ worker ได้ทดสอบแพ็กเกจขนาด 34,269,921 bytes ซ้ำบนเครื่องเดิม โหมด bundled ใช้ 2.564 วินาที และ external ใช้ 2.058 วินาที ทั้งสองอ่านชื่อ workspace ภาษาไทยจาก DOM และ worker ออกด้วย code 0

## ทำซ้ำ

บนเครื่อง build ใช้ Python launcher ที่เลือก 3.12 ได้, Node.js กับ npm และ Windows x64 ที่ติดตั้ง WebView2 Runtime จาก repository root รัน

```powershell
.\spikes\desktop\build.ps1
py -3.12 spikes/desktop/smoke.py
```

ถ้ามี project venv ให้ทดสอบ worker ด้วย executable เดียวกับที่เตรียมไว้สำหรับ inference ด้วย

```powershell
py -3.12 spikes/desktop/smoke.py --runtime .venv/Scripts/python.exe
```

หากเครื่อง build จำกัดการรัน PowerShell scripts ให้รันคำสั่งใน [build.ps1](../../spikes/desktop/build.ps1) ตามลำดับในโฟลเดอร์ของ spike คำสั่ง CI ใช้ smoke script เดียวกันทดสอบทั้ง bundled worker และ external worker และต้องมี Windows session ที่สร้าง hidden WebView2 window ได้

สำหรับการทดสอบด้วยตนเอง เปิด `spikes/desktop/dist/LayerDeltaSpike/LayerDeltaSpike.exe` แล้วเลือกโฟลเดอร์ผ่าน native dialog เมื่อยกเลิกการเลือก launcher จบก่อนสร้าง worker ส่วน smoke script ป้อนเส้นทางผ่าน `--workspace`

## ข้อจำกัดและจุดตรวจถัดไป

- การทดสอบนี้ใช้ temporary copy บนเครื่องพัฒนาที่มี WebView2 ติดตั้งอยู่ ยังต้องยืนยันวิธีตรวจหรือติดตั้ง prerequisite บน Windows สะอาด และทดสอบการเลือก/ยกเลิก native dialog ด้วยตนเองก่อนผ่าน G1
- ระหว่างทดลองครั้งแรกพบ hidden WebView2 รอ startup event เกิน 20 วินาทีหนึ่งครั้ง รอบ staging ที่บันทึกในตารางผ่านทั้งแปดรัน ให้เก็บ phase, exit code และ timeout แยกกันใน M2 เพื่อวินิจฉัยการเริ่มต้นที่ช้าบนเครื่องผู้ใช้
- งานใน worker ของ spike เป็นการอ่านชื่อ workspace ตัววัดเวลานี้ครอบคลุม shell, worker และ React renderer การโหลดโมเดลและ inference CPU/CUDA ใช้หลักฐานจาก reference probe และ runner ใน M0/M1 ส่วน package ที่รวม workflow จริงต้องวัดซ้ำใน M3/M4
- เมื่อเชื่อม application service ใน M2 ให้ server ตรวจ session, project scope และ actor ตามสัญญา API ตัว launcher จำกัดเวลารอสัญญาณพร้อมจาก worker ไว้ 20 วินาที เมื่อเกินเวลาจะปิด pipe และรอ worker ไม่เกินอีก 5 วินาทีก่อนบังคับหยุด
