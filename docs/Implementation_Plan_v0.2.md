# LayerDelta v0.2 — แผนพัฒนา: Agent weight inspection, session trace และ benchmark

ต่อจาก [แผน v0.1](Implementation_Plan_v0.1.md) ซึ่งส่งมอบ workflow, Agent API และ release candidate แล้ว ข้อกำหนดเต็มของงานนี้อยู่ใน [LayerDelta_Agent_Extension.md](LayerDelta_Agent_Extension.md) เอกสารนี้ระบุว่า repository มีอะไรแล้ว ต้องเพิ่มอะไร และตรวจรับอย่างไร ผลว่า agent ทำได้ดีขึ้นเท่าใดต้องมาจาก benchmark ที่รันจริงเท่านั้น

## สถานะเทียบกับ v0.1

| ความสามารถ | สถานะ | หมายเหตุ |
|---|---|---|
| `inspect → propose → run → status → compare`, `get_changes`, `get_budget` | มีแล้ว | `layerdelta.client.AgentClient` |
| Weight intervention แบบ scale/zero หนึ่งจุดต่อ recipe | มีแล้ว | G3 |
| งบ, idempotency, scoped token, holdout ที่ Agent เข้าไม่ได้, audit | มีแล้ว | G8, G9 |
| Weight preview | **บางส่วน** | `POST /previews` คืนสถิติ kernel ราย output channel พ่วงกับ activation และต้องมี `sample_id` ยังไม่มีค่า kernel รายตำแหน่ง, ช่วง input-channel, pagination, fingerprint/axes หรือ heatmap |
| ประวัติการทดลอง | **บางส่วน** | `jobs`, `recipes`, `holdout_audit` ครอบการ submit และผล run ไม่มี session ID, case/arm, event การอ่าน และ timeline |
| Benchmark สอง arm | **บางส่วน** | `tools/verify_m3.py` ใช้ policy แบบ deterministic หนึ่งโจทย์ arm ต่างกันที่ activation statistics ไม่ใช่ weight preview และยังไม่มี LM |
| Provider adapter และนโยบายส่งข้อมูลออก | ยังไม่มี | นโยบายมีใน [agent.md](../agent.md) แต่ไม่มีโค้ดให้ตรวจ |

## ข้อควรระวังก่อนเริ่ม

1. **ปัญหาความแฟร์ของ arm ควบคุม:** preview เดิมคืนสถิติ kernel ทุกครั้ง ถ้าไม่แยกก่อน arm ที่ "ปิด weight preview" ยังเห็นสถิติ weight ได้ ต้องแก้ใน M5.1 ก่อนทำ benchmark
2. **ช่องรั่วทางอ้อม:** ตรวจ report ของ weight run ซึ่งบันทึกสถิติ kernel ก่อนและหลังแก้ (`_patch_kernel`) ว่า arm ควบคุมอ่านได้หรือไม่ ถ้าอ่านได้ ต้องซ่อนจาก Agent ตาม capability
3. **โจทย์สาธิต:** fixture ที่ตรึงไว้ใช้ภาพเบลอที่ label จริงเป็น 9 เอกสารเสริมต้องการ 8→3 ตัดสินใจใน M5.0 ว่าจะเปลี่ยนโจทย์หรือทำให้เป็นพารามิเตอร์ การเปลี่ยนต้องไม่ทำให้ recipe hash และผลอ้างอิงของ v0.1 เปลี่ยน
4. **v0.1 ไม่ต้องรอ:** ทุกอย่างที่เพิ่มต้องเป็น additive ต่อ contract เดิม caller เดิมที่ส่ง `sample_id` ต้องทำงานเหมือนเดิม

## ลำดับงาน

### M5.0 · ตรึงโจทย์และข้อตกลง

- เลือกโจทย์ (9 ตาม fixture เดิม หรือ 8→3) จาก baseline จริงบน exploration และดูว่ามีตัวอย่างพอ บันทึกเหตุผลก่อนเขียน benchmark
- กำหนดล่วงหน้า: metric หลัก, guardrail และเกณฑ์, งบ run, งบ token, จำนวนคู่ session, frontier model และ config
- ตัดสินใจนโยบายข้อมูลที่ส่งให้ provider ภายนอกได้ (ค่าเริ่มต้นคือไม่ส่งภาพและ tensor)

**ผ่านเมื่อ** มีเอกสารตัดสินใจที่ commit แล้วและ fixture เดิมยัง pass

### M5.1 · Weight preview และสิทธิ์ weight

1. **Contract:** เพิ่ม typed tool `preview_weights` และ endpoint ที่รับ checkpoint ID, target ID, output channel และช่วง input-channel ตาม mapping `[C_out, C_in/groups, K_h, K_w]` คืนชื่อ parameter, shape, dtype, checkpoint fingerprint, tensor axes, ช่วง index ที่คืนจริง, สถิติของ slice และค่าตัวเลขเรียงตาม axes พร้อม cursor เมื่อเกินเพดาน ไม่ต้องเลือก sample
2. **แยกสิทธิ์:** เพิ่ม capability `weights:read` ที่ผูกกับ token/session ตรวจที่ backend ก่อน resolve object ถอดสถิติ kernel ออกจาก activation preview และจาก report ที่ Agent อ่าน เมื่อไม่มีสิทธิ์ Human UI ยังเห็นเหมือนเดิม
3. **Desktop:** วาด kernel heatmap จากข้อมูลชุดเดียวกับที่ Agent ได้ ให้คนกับ Agent อ้างตำแหน่ง kernel เดียวกัน
4. **Client:** เพิ่มเมธอดใน `AgentClient` และ deterministic stub

**ผ่านเมื่อ (G11)** ค่าและ axes ตรงกับ tensor ต้นทางทุก target ตรวจ channel แรก/ท้ายและ slice ที่ต่อหน้า, target ผิดหรือขอเกินขอบเขตถูกปฏิเสธ, preview เดิมที่ส่ง `sample_id` ได้ผลเหมือนเดิม, token ที่ไม่มี `weights:read` ไม่เห็นค่าหรือสถิติ weight จาก endpoint, report หรือ catalog ใดเลย

### M5.2 · Session trace และ timeline

1. เพิ่มตาราง event ต่อการเรียก tool ที่ backend รับรู้: session ID, ลำดับและเวลา, operation, request ID, status หรือ error code, checkpoint ID/fingerprint, target/channel/ช่วงที่อ่าน และ recipe/run/baseline ID ผูก session กับ benchmark case และ arm
2. Preview event อ้างถึง slice ที่คืนโดยไม่คัดลอก tensor ลง log Mutation event เชื่อม propose → run → ผล และแสดงสถานะ proposed/queued/succeeded/failed/cancelled การส่งซ้ำด้วย key เดิมต้องเชื่อมกับ run เดิมและหักงบครั้งเดียว
3. ตัด token, header ยืนยันตัวตน, ภาพ และ prompt เต็มออกจาก log
4. Timeline ต่อ session ที่เปิดจาก tool call ไปยัง recipe และผล run ได้ ใช้ schema เดียวกันทั้งสอง arm Dashboard ข้ามโปรเจกต์และ model registry อยู่นอกสโคป

**ผ่านเมื่อ (G12)** ตรวจย้อนจาก weight preview ไปยัง recipe และผล run ได้, retry ไม่สร้าง run หรือหักโควตาซ้ำ, ค้นใน log ไม่พบ token หรือ tensor เต็ม, restart แล้ว trace ยังอยู่ครบ

### M5.3 · Benchmark runner และรายงาน

1. ทำ runner ที่สร้าง session ต่อ (case, arm, คู่) บังคับสิทธิ์ weight ที่ backend และจำกัดเครื่องมือทางอ้อมใน runner สลับลำดับรันเพื่อลดผลจากลำดับ และจับคู่ session ตามโจทย์และจุดเริ่ม
2. ให้ทั้งสอง arm ใช้ checkpoint, sample, feedback, งบรัน และงบ agent เท่ากัน ต่างกันเฉพาะ `weights:read`
3. แต่ละ session ส่งสูตรสุดท้ายหนึ่งสูตรก่อนเปิด holdout แล้ว freeze recipe hash ผู้ประเมินรัน holdout ผ่านสิทธิ์ human บนภาพชุดเดียวกัน
4. รายงานแยก session: จำนวนตัวอย่าง, ผลต่าง metric หลัก, accuracy ภาพปกติ, จำนวนที่แก้ถูกและเสียเพิ่ม, runs/tokens/เวลา สรุปผลต่างระหว่าง arm แบบจับคู่พร้อมความไม่แน่นอน และใช้ trace ตรวจว่า arm ทดลองเปิดดูอะไรก่อนเลือกแก้
5. เพิ่ม provider adapter (local และ frontier) ที่ส่งเฉพาะข้อมูลที่นโยบายอนุญาต พร้อมทดสอบ payload ด้วย provider จำลอง
6. รันครบวงจรด้วย agent แบบ deterministic บน fixture ก่อน แล้วจึงใช้ frontier model จริง

**ผ่านเมื่อ (G13)** ทั้งสอง arm แก้และประเมินได้เท่ากัน, arm ควบคุมอ่าน weight ไม่ได้ทุกช่องทาง, agent ทั้งสองเข้า holdout ไม่ได้, รันครบวงจรบน fixture ได้ผลซ้ำ และรายงานระบุขอบเขตการตีความ (CNN, checkpoint, งาน OCR และโมเดลที่ทดสอบ) การทดลองนี้วัดประโยชน์ส่วนเพิ่มของการเห็น weight ภายใต้ operation ที่จำกัด ไม่ได้วัดมูลค่าของ LayerDelta ทั้งชุด

## ตาราง gate เพิ่มเติม

| Gate | หลักฐาน | จุดตรวจ |
|---|---|---|
| G11 Weight preview | ค่า/axes ตรงกับ tensor, scope และ backward compatibility | M5.1 |
| G12 Trace | timeline ย้อนจาก preview ถึงผล run, ไม่มี secret/tensor ใน log, retry ไม่ซ้ำ | M5.2 |
| G13 Benchmark | arm เท่ากันยกเว้นสิทธิ์ weight, ไม่รั่ว, ไม่เข้า holdout, รายงานแบบจับคู่ | M5.3 |

## นอกสโคป

MCP bridge (ทำหลัง HTTP contract นี้เสถียร โดย map tools ตารางเดียวกัน), การแก้หลายตำแหน่ง, architecture อื่น, benchmark ของ workflow ปกติของผู้ใช้ และ dashboard ข้ามโปรเจกต์
