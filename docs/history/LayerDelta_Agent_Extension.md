# LayerDelta — Agent weight inspection, experiment trace, and benchmark

> **Status:** parked research spec. The weight-access split and trace are now v0.2 tasks T4/T5, and the benchmark is scheduled for v0.5 across several model families. See [Roadmap](../roadmap/Roadmap.md).

เอกสารเสริมสำหรับ coding agent · ข้อกำหนดที่เสนอสำหรับการอัปเดต repository

## เป้าหมาย

ให้ agent เปิด checkpoint เดียวแล้วสำรวจ weight ของ CNN ตาม layer และ channel เพื่อใช้ข้อมูลประกอบการเลือกแก้ weight จากนั้นวัดผลของการแก้บนตัวอย่างชุดเดียวกับ baseline และย้อนดูได้ว่า agent ใช้ข้อมูลอะไรและทดลองอะไร เป้าหมายวิจัยคือทดสอบว่าการเห็น weight ช่วยให้ agent เลือกการแก้ที่ให้ผลดีกว่าภายใต้งบเท่ากันหรือไม่

อิงสัญญาจาก [Agent Integration](../guides/agent-integration.md), [API](../reference/api.md) และ [Architecture](../architecture/Architecture.md) ให้ตรวจชื่อไฟล์และ implementation จริงใน repo ก่อนแก้

## 1. เปิดให้ agent สำรวจ weight

- เพิ่ม typed tool `preview_weights` เป็นชื่อเสนอใน HTTP wrapper เดิม ใช้ catalog จาก `inspect` เพื่อเลือก checkpoint, target และ output channel ให้ wrapper ใช้ preview job และ status contract เดิม โดยเพิ่มผลแบบ structured สำหรับ Conv kernel slice
- รับ checkpoint ID, target ID, output channel และช่วง input-channel ตาม mapping ของ adapter คืนชื่อ parameter, shape, dtype, checkpoint fingerprint, tensor axes, ช่วง index ที่คืนจริง, สถิติของ slice และค่า kernel แบบตัวเลขเรียงตาม axes ที่ระบุ ให้เรียกช่วงถัดไปได้เมื่อ slice ใหญ่กว่าเพดาน
- สำหรับ Conv2d ให้ใช้ mapping เดิม `[C_out, C_in/groups, K_h, K_w]` โดยเลือก output channel ก่อน แล้วจึงเลือก input-channel slice ภายใน group ค่าที่คืนต้องมาจาก checkpoint ที่ระบุจริง ใช้ข้อมูลชุดเดียวกันวาด heatmap ใน UI เพื่อให้คนกับ agent อ้างถึง kernel ตำแหน่งเดียวกัน
- ตรวจสิทธิ์ project/session, target, index และเพดาน response ที่ backend ใช้ข้อมูลจาก catalog แทนพาธไฟล์จาก agent การอ่าน weight ใช้ checkpoint โดยตรง หาก preview เดิมบังคับ sample ID ให้คงความเข้ากันได้กับ caller เดิมและเพิ่มสัญญาที่รองรับ weight preview โดยไม่ต้องเลือกภาพ
- เปิด backend จาก Desktop หรือ headless CLI ใช้ wrapper เดียวกับ local หรือ frontier LM ข้อมูล weight และภาพอยู่ในเครื่องเป็นค่าเริ่มต้น ส่งให้ frontier provider ได้เมื่อผู้ใช้อนุญาตตามขอบเขตที่กำหนด ทำ HTTP integration ให้ผ่านก่อนเพิ่ม MCP bridge ที่เรียก API เดิม
- คง flow ทดลองเดิม `inspect → propose → run → status → compare` ใช้ `get_changes` ดูตัวอย่างที่ทำนายเปลี่ยน และ `get_budget` ดูงบ คำว่า `compare` ใน flow นี้หมายถึงผลทำนาย baseline และ intervention บนตัวอย่างชุดเดียวกัน ใช้โหมด weight สำหรับ benchmark นี้และใช้ operation ที่ adapter รองรับ เช่น scale หรือ zero หนึ่ง intervention ต่อ recipe

## 2. เก็บประวัติการใช้ tool และการทดลอง

ใช้ SQLite และข้อมูล recipe/run ที่มีอยู่ เพิ่ม event ต่อการเรียก tool ที่ backend รับรู้ โดยเก็บ session ID, เวลาและลำดับ, ชื่อ operation, request ID, สถานะหรือ error code, checkpoint ID/fingerprint, target/channel หรือช่วงที่อ่าน และ recipe/run/baseline ID ตามที่เกี่ยวข้อง ใน benchmark ให้ผูก session กับ benchmark case และ arm เพื่อให้ตรวจสอบเงื่อนไขย้อนหลังได้

- Preview event ระบุว่าขอช่วงใดและส่งผลสำเร็จหรือไม่ อ้างถึง checkpoint และ slice ที่คืนโดยไม่คัดลอก tensor ทั้งก้อนลง log
- Mutation event เชื่อมสูตรที่เสนอ งานที่รับเข้าคิว และผลรัน แสดงให้ชัดว่างานถูกเสนอ ถูกสั่งรัน สำเร็จ ล้มเหลว หรือยกเลิก การส่งซ้ำด้วย idempotency key เดิมอาจปรากฏในประวัติคำขอ แต่ต้องเชื่อมกับ run เดิมและใช้งบครั้งเดียว
- ตัด token, header สำหรับยืนยันตัวตน, ภาพ และ prompt เต็มออกจาก log ใช้ recipe และ report เป็นแหล่งรายละเอียดของ operation และ metrics
- MVP ใช้ timeline ต่อ session ที่เปิดจาก tool call ไปยัง recipe และผล run ได้ ใช้ schema เดียวกันทั้งสอง arm ส่วน dashboard ข้ามโปรเจกต์และ model registry อยู่นอกสโคปนี้

## 3. Benchmark ผลของการเห็น weight

ใช้การทดลองสอง arm ที่ต่างกันเฉพาะสิทธิ์เรียก weight preview

| เงื่อนไข | เห็น weight | ควบคุม |
|---|---|---|
| เครื่องมือร่วม | Catalog, แก้ weight, รัน, ดู metrics และ prediction changes | ชุดเดียวกัน |
| การเข้าถึง weight | อ่านสถิติและค่า kernel ตามขอบเขต | ปิด weight preview และทางเข้าถึงค่าดิบหรือสถิติ weight อื่น |
| จุดเริ่มและงบ | Architecture และ checkpoint เดียวกัน, frontier model และ config เดียวกัน, โจทย์และ split เดียวกัน, งบรันและงบ agent เท่ากัน | เงื่อนไขเดียวกัน |

1. เริ่มด้วย CNN OCR ที่อ่าน 8 เป็น 3 ในภาพเบลอ กำหนด metric หลักเป็นอัตราการทำนาย 3 ในกลุ่มภาพเบลอที่ label จริงคือ 8 และใช้ accuracy ของภาพปกติเป็น guardrail กำหนดเกณฑ์ guardrail, งบ run, งบ token และจำนวน session ล่วงหน้า
2. ให้ทั้งสองฝั่งใช้ sample และ feedback จาก exploration เหมือนกัน บังคับสิทธิ์ preview ที่ backend และจำกัดเครื่องมือทางอ้อมใน benchmark runner จับคู่ session ตามโจทย์และจุดเริ่มต้น ใช้ session อิสระหลายคู่และสลับลำดับรันเพื่อลดผลจากลำดับ
3. ให้แต่ละ session ส่งสูตรสุดท้ายหนึ่งสูตรก่อนเปิดผล holdout แล้ว freeze recipe hash ให้ผู้ประเมินรัน holdout ผ่านสิทธิ์ human ใช้ภาพชุดเดียวกันสำหรับ baseline และทุกสูตรที่นำมาเทียบ
4. รายงานผลแยก session พร้อมจำนวนตัวอย่าง ผลต่าง metric หลัก, accuracy บนภาพปกติ, จำนวนที่แก้ถูกและจำนวนที่เสียเพิ่ม รวมถึง runs, tokens และเวลา สรุปผลต่างระหว่าง arm แบบจับคู่และรายงานความไม่แน่นอน ใช้ trace ตรวจว่าฝั่งทดลองเปิดดูอะไร ก่อนเลือกแก้ตรงไหน
5. ตีความผลตามขอบเขต CNN, checkpoint, งาน OCR และ frontier model ที่ทดสอบ การทดลองนี้วัดประโยชน์ส่วนเพิ่มของการเห็น weight สำหรับการแก้แบบจำกัด operation หากต้องการวัดมูลค่าของ LayerDelta ทั้งชุด ให้ทำ benchmark แยกกับ workflow ปกติของผู้ใช้

## ลำดับลงมือและเกณฑ์รับงาน

1. ต่อ weight preview เข้ากับ wrapper และ adapter เดิม ตรวจว่า agent client แบบ deterministic เปิดดู slice ของ checkpoint เดียวได้ ค่าและ axes ตรงกับ tensor ต้นทาง และ target ผิดหรือขอเกินขอบเขตถูกปฏิเสธ
2. เพิ่ม session trace และ timeline ตรวจย้อนจาก weight preview ไปยัง recipe และผล run ได้ ตรวจว่า retry ไม่สร้าง run หรือหักโควตาซ้ำ และไม่มี token หรือ tensor เต็มใน log
3. ทำ benchmark runner และรายงานผล ตรวจว่าทั้งสอง arm แก้และประเมินได้เท่ากัน arm ควบคุมอ่าน weight ไม่ได้ และ agent ทั้งสองฝั่งเข้าถึง holdout ไม่ได้ จากนั้นลองรันครบวงจรบน fixture ก่อนใช้ frontier model จริง

งานในเอกสารนี้คือเพิ่มความสามารถและเครื่องมือประเมิน ผลว่า agent ทำได้ดีขึ้นเท่าใดต้องมาจาก benchmark ที่รันจริง
