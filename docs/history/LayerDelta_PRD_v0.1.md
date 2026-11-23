# LayerDelta — Product Requirements Document

> **Status:** v0.1 scope, delivered as an internal build. The product direction has since broadened to model-agnostic weight-space tooling; current milestones and status are in [Roadmap](../roadmap/Roadmap.md). Examples below that use 8→3 predate the frozen fixture, whose demonstration slice is blurred true 9 (most frequent error 9→4).

M2 alpha exposes a two-block MNIST workbench through the Windows desktop launcher and loopback headless service. The explorer requests the original image, one post-ReLU channel map, and kernel statistics for a paired exploration sample. Each run records the actual device. Final evaluation and weight-child export require a human session and explicit confirmation. The executable payloads and response units are documented in the [HTTP contract](../reference/api.md).

| รายการ | รายละเอียด |
|---|---|
| ชื่อแอป / repo และ CLI | LayerDelta / `layerdelta` |
| เวอร์ชัน | v0.1 · ฉบับร่าง สำหรับวางแผนพัฒนาและตรวจรับ |
| วันที่ | 22 พฤศจิกายน 2026 |
| รูปแบบผลิตภัณฑ์ | Local-first open-source workbench สำหรับทดลองกับโมเดล deep learning |
| กลุ่มผู้ใช้หลัก | นักศึกษา นักวิจัย และ ML Engineer ที่ต้องการวิเคราะห์พฤติกรรมโมเดล |
| เอกสารต้นทาง | [LayerDelta Open Source Proposal](LayerDelta_Open_Source_Proposal.md) |

## 1. ปัญหาและคุณค่าผลิตภัณฑ์

เมื่อ CNN อ่านภาพเลข 8 เป็น 3 บ่อยขึ้นในภาพเบลอ ผู้ทดลองเปิด confusion matrix เพื่อเห็นรูปแบบข้อผิดพลาดได้ แต่การทดสอบว่า output channel ใดมีส่วนต่อความผิดพลาดต้องเขียนโค้ดหลายช่วง ตั้งแต่ระบุ layer และจุด hook ไปจนถึงรันข้อมูลคู่เดิม คำนวณผลรายกลุ่ม และเก็บหลักฐานว่ารันแต่ละครั้งแก้อะไร

LayerDelta ให้ผู้ใช้เลือกเป้าหมายที่ระบบระบุได้ ตั้งสมมติฐาน รันการเปลี่ยนแปลงหนึ่งจุด แล้วเห็นว่าคำทำนายเปลี่ยนไปในกลุ่มเป้าหมายและกลุ่มอื่นอย่างไร Desktop UI และ Agent API ใช้กติกาตรวจสอบ สูตรทดลอง และผลวัดชุดเดียวกัน ผู้รับสูตรจึงตรวจสอบการตั้งค่าและรันซ้ำได้เมื่อมี artifacts ตรงรุ่น

**ผลลัพธ์หลักของ v0.1** ผู้ใช้เริ่มจาก CNN ตัวอย่าง แล้วส่งต่อสูตรพร้อมผล baseline และ intervention ให้คนหรือ Agent อีกรายรันซ้ำได้ครบหนึ่งรอบ การเพิ่ม accuracy เป็นผลการทดลองที่อาจเกิดขึ้นได้ รายงานผลที่ดีขึ้น แย่ลง และไม่มีการเปลี่ยนแปลงต้องตรวจสอบย้อนหลังได้เท่ากัน

## 2. ผู้ใช้และงานที่ต้องทำ

| ผู้ใช้ | งานหลัก | หลักฐานว่างานเสร็จ |
|---|---|---|
| นักศึกษาและผู้สอน | ดู filter และ activation ที่สัมพันธ์กับ channel เดียวกัน แล้วทดลองเปลี่ยนค่า | อธิบายได้ว่าแก้ที่ตำแหน่งใดและชี้ตัวอย่างคำทำนายที่เปลี่ยน |
| นักวิจัยและ ML Engineer | ตรวจ failure slice และหาผลกระทบข้ามกลุ่ม | เปรียบเทียบผลบนตัวอย่างคู่เดิม พร้อมจำนวนตัวอย่างและผลเสียที่เกิดขึ้น |
| ผู้พัฒนา Agent | ให้ Agent สำรวจเป้าหมายและทดสอบสมมติฐานภายใต้งบ | ใช้ structured API ครบจาก inspect ถึง compare และตรวจย้อนหลังได้ว่า Agent ใช้งบเท่าไร |
| ผู้ร่วมพัฒนาโอเพนซอร์ส | ทำซ้ำผลหรือเตรียมรองรับโมเดลใหม่ | เปิดตัวอย่าง ทำตามคู่มือ และเข้าใจสัญญาของ model adapter |

## 3. ขอบเขตและลำดับความสำคัญ

P0 คือฟังก์ชันที่ต้องผ่านก่อนเปิด v0.1 ส่วน P1 คือรายการเพิ่มคุณค่าที่จัดลำดับได้หลัง workflow หลักผ่านการตรวจรับ

| ความสำคัญ | ขอบเขต |
|---|---|
| P0 | PyTorch CNN จำแนกเลข 0–9 หนึ่ง architecture, checkpoint อ้างอิงหนึ่งชุด, ภาพปกติ/เบลอที่สร้างซ้ำได้ |
| P0 | Desktop app ในเครื่อง, แผนผัง layer, รายการ output channels, weight statistics และ activation map สำหรับภาพที่เลือก |
| P0 | Activation intervention และ filter weight edit แบบ zero/scale ครั้งละหนึ่งเป้าหมาย |
| P0 | Baseline, before/after, ผลรายกลุ่ม, ประวัติรัน, สูตร JSON, ส่งออกผลวัดและ child checkpoint จาก weight edit |
| P0 | API inspect/propose/run/compare ที่ใช้ backend เดียวกับ UI และจำกัดงบ Agent |
| P1 | กราฟเปรียบเทียบหลายรัน, การค้นประวัติขั้นสูง และตัวช่วยสำรวจหลาย channel |
| รุ่นหลัง v0.1 | โมเดลทั่วไปและ custom code, arbitrary tensor slices, multi-target edits, retraining, Transformer, OCR recognizer, training intervention และการเผยแพร่ checkpoint อัตโนมัติ |

### ข้อเสนอสำหรับสเปกเริ่มต้น

ใช้ **MNIST และ CNN ขนาดเล็กสอง Conv blocks** เป็น fixture ตั้งต้นเพื่อให้มีตัวเลข 0–9 และมีหลาย output channels ให้สำรวจ กำหนด layer ที่ UI เปิดให้แก้ผ่าน adapter ของโมเดลที่ติดมากับโปรแกรม เลือก topology จริงและ freeze checkpoint ใน PoC ค่า scale ที่เสนอคือค่าจำกัดในช่วง [0, 2] แบบรวมปลายทั้งสองด้าน โดย 1 ใช้ตรวจ identity และ 0 ให้ผลแบบเดียวกับ zero ที่ตำแหน่งเดียวกัน สถาปัตยกรรมจริง ตัวเลขงบ Agent และ Apache-2.0 ยังเป็นข้อเสนอ

Desktop packaging และ headless mode เป็น P0 สำหรับ v0.1. MCP เป็น integration หลัง HTTP tool contract ผ่านการตรวจรับและไม่บล็อก release. OS เป้าหมายและ desktop shell ให้ยืนยันจาก packaging spike ใน M0–M1.

## 4. Workflow สำหรับตรวจรับ

1. ผู้ใช้เปิดโปรเจกต์ตัวอย่าง ระบบแสดง architecture, fingerprint ของ checkpoint และเวอร์ชัน dataset
2. ผู้ใช้เลือก layer และ channel แล้วดู kernel shape, weight statistics และ activation map ของภาพที่เลือก ระบบแสดงตำแหน่ง hook เช่น output หลัง ReLU ของ block
3. ผู้ใช้รัน baseline บน exploration split แล้วเลือกดู slice ภาพเบลอและตัวอย่างที่ baseline อ่าน 8 เป็น 3
4. ผู้ใช้สร้างสูตร เช่น คูณ activation ของ channel ที่เลือกด้วย 0.5 และบันทึกสมมติฐานว่าการแก้จะเปลี่ยนจำนวนข้อผิดพลาด 8 → 3 ในภาพเบลอ
5. ผู้ใช้ดูผล paired comparison และตรวจผลบนภาพปกติ คลาสอื่น และตัวอย่างที่แก้ถูกหรือกลายเป็นผิด
6. ผู้ใช้สร้างสูตร weight edit แยกเพื่อคูณ kernel ของ channel เดียวกัน แล้วตรวจผลประเมินอีกครั้ง
7. ผู้ใช้ส่งออก recipe และ metrics ให้ Agent หรือผู้ร่วมงานรันซ้ำ จากนั้นเลือกสูตรที่จะประเมินกับ holdout และอนุมัติการส่งออก child checkpoint ผ่าน UI หากต้องการ

## 5. ข้อกำหนดเชิงฟังก์ชัน

| ID | ฟังก์ชัน P0 | พฤติกรรมและเกณฑ์ตรวจรับ |
|---|---|---|
| FR-01 | เปิดโปรเจกต์ตัวอย่าง | Quickstart เปิด CNN ชุดอ้างอิงได้จนถึงหน้า inspect แสดง model ID, architecture revision, checkpoint hash และ dataset revision ตรงกับ manifest |
| FR-02 | โหลด checkpoint | รับ tensor-only checkpoint ของ architecture ที่รองรับ ตรวจขนาดไฟล์ รายชื่อ tensor, shape, dtype และ schema ก่อน inference ไฟล์ไม่ตรงสเปกให้ข้อความระบุจุดผิด โดยโปรเจกต์เดิมยังใช้งานได้ |
| FR-03 | สำรวจโมเดล | แสดงลำดับ layer และ output channels จาก adapter ระบุ index แบบเริ่มที่ 0, tensor shape, kernel statistics และ activation map ที่ผูกกับ sample ID จุดที่สำรวจได้อย่างเดียวต้องมีสถานะ read-only |
| FR-04 | Baseline | ใช้ checkpoint ต้นทาง, model eval mode, preprocessing และ sample set ตาม manifest เก็บคำทำนายต่อ sample ID กับ metrics การนำผลเดิมมาใช้ต้องจับคู่ fingerprint ของเงื่อนไขการประเมินครบ |
| FR-05 | นิยามการทดลอง | ฟอร์มและ API รับ hypothesis, target ID, mode, operation, value และ evaluation config ตาม schema เดียวกัน การ propose สำเร็จสร้าง recipe ที่อยู่สถานะพร้อมรันและยังไม่มีผลวัด |
| FR-06 | Activation intervention | รองรับ zero หรือ scale ที่ output channel ณ hook ที่ประกาศ ใช้กับทุก sample ใน run เกณฑ์ fixture ต้องยืนยันว่า channel ที่เลือกเปลี่ยนตามสูตรและ channel อื่น ณ จุดเดียวกันคงค่าเดิม |
| FR-07 | Weight edit | zero หรือ scale เฉพาะ kernel slice ของ output channel ที่เลือก โดยแสดงชื่อ tensor, slice และสถิติก่อน/หลัง ตรวจค่า finite หลังแปลงและบันทึก hash ของ parent checkpoint |
| FR-08 | จัดการรัน | แสดงสถานะและความคืบหน้าที่สังเกตได้ มี cancel และ retry รันถัดไปบน baseline เดิมต้องให้ผลเดิมหลังรันก่อนหน้าสำเร็จ ล้มเหลว หรือถูกยกเลิก ตรวจ hash checkpoint ต้นทางประกอบ |
| FR-09 | เปรียบเทียบ | แสดง baseline/intervention และ delta บนกลุ่มเดียวกัน พร้อม sample count, confusion matrices และรายการตัวอย่างที่ prediction เปลี่ยน ฝั่ง UI และ API ให้ผลตรงกันจาก run เดียว |
| FR-10 | สูตรและประวัติ | เปิดประวัติจาก run ID และ export/import สูตร JSON ได้ เมื่อ fingerprint หรือ schema ไม่ตรง ให้ระบุ dependency ที่ต้องใช้ก่อนเปิดให้รัน |
| FR-11 | ส่งออกผล | ส่งออก recipe และ metrics JSON สำหรับทั้งสองโหมด แสดงจำนวนตัวอย่าง ขอบเขตข้อมูล และ provenance ที่จำเป็นต่อการตีความ |
| FR-12 | ส่งออก checkpoint | ผู้ใช้ยืนยันจาก weight run ที่สำเร็จ ระบบสร้าง child artifact ที่มี parent hash, recipe ID และ child hash โดยไม่ทับชื่อเดิม การโหลด child กลับแล้วประเมินแบบไม่มี patch ต้องให้ผลตรงกับ weight run ต้นทาง |
| FR-13 | Agent API | Agent อ่าน target catalog เสนอสูตร รัน และอ่าน comparison ภายใต้ credential และงบที่กำหนด ฝั่ง backend ตรวจสิทธิ์ export และ final evaluation แยกจากสิทธิ์รัน exploration |
| FR-14 | Final evaluation | ผู้ใช้ยืนยันสูตรที่ freeze แล้วก่อนเริ่ม ระบบบันทึกการเปิดใช้ holdout พร้อม recipe hash และคืนผลที่เปรียบเทียบได้ทั้ง overall, normal และ blurred พร้อม N |

### ความหมายของการแก้ไข

| โหมด | เป้าหมาย | การทำงาน | Artifact |
|---|---|---|---|
| Activation | Output tensor ของจุด hook ที่ adapter ประกาศ | เปลี่ยน slice ของ channel ตาม index และคง shape/dtype ตลอด forward | Recipe และผลรัน |
| Weight | Kernel ของ Conv output channel ที่ adapter map ให้ | เปลี่ยน slice ของ weight tensor บน working model โดย scope ของ v0.1 ใช้ kernel เท่านั้น | Recipe, ผลรัน และ child checkpoint เมื่อผู้ใช้ยืนยัน |

ตำแหน่งก่อนหรือหลัง activation function เป็นส่วนหนึ่งของ target identity ตัวอย่างเช่น การคูณ kernel ก่อน bias และ ReLU กับการคูณ feature map หลัง ReLU คือคนละการทดลอง UI ต้องแสดงตำแหน่งนั้นขณะเลือกเป้าหมายและขณะอ่านผล

## 6. ข้อมูลและวิธีประเมิน

### ชุดข้อมูลและการแบ่ง split

- แบ่ง train, exploration และ final holdout ด้วย source sample ID ก่อนสร้างภาพแปรรูป เก็บรายการสมาชิกและ hash ไว้ใน manifest
- สร้าง normal และ blurred จากภาพต้นทางตาม transform ที่มี revision และ parameter คงที่ ใช้ ID ที่แยก source image กับ condition ได้
- ทุก comparison จับคู่ baseline และ intervention ตาม sample ID, split และ condition แสดง N ของแต่ละกลุ่ม รวมทั้งนิยามน้ำหนักของ overall หากรวมสอง conditions
- Slice อย่าง baseline อ่าน 8 เป็น 3 ให้ตรึงสมาชิกจาก baseline เดิมตลอด comparison ส่วน primary holdout slice ใช้เงื่อนไขที่กำหนดไว้ล่วงหน้า เช่น label 8 และ blurred
- หน้า exploration ปิดการเข้าถึง holdout ผ่าน Desktop UI และ Agent API ตามค่าเริ่มต้น สูตรที่จะรายงานเป็นผลยืนยันต้องบันทึกก่อนเปิด holdout การดู holdout หลายรอบต้องถูกรายงานเมื่อสรุปผล

### เมตริกและการแสดงผล

| ตัววัด | นิยามที่ใช้ |
|---|---|
| Accuracy | จำนวนทายถูกหารด้วย N ของกลุ่มเดียวกัน แสดง baseline, intervention และส่วนต่างเป็น percentage points |
| Confusion matrix | แถวเป็น ground truth คอลัมน์เป็น prediction ใช้ลำดับคลาส 0–9 เหมือนกันทั้งคู่ |
| ผลรายคลาสและสภาพภาพ | Accuracy ของตัวอย่างที่ label ตรงคลาสนั้น แยก normal/blurred ตามตัวกรองที่กำหนดและแสดง N |
| Prediction changed | จำนวน sample IDs ที่ predicted class เปลี่ยน แยกรายการผิด→ถูก ถูก→ผิด และผิด→ผิดคนละคลาส |
| กลุ่มว่าง | แสดง N = 0 และ metric เป็น null พร้อมเหตุผล เพื่อให้ export และ UI ตีความตรงกัน |

กรณีตัวอย่างจากกลุ่มเล็ก ให้ผู้อ่านเห็นจำนวนจริงควบคู่กับเปอร์เซ็นต์ รายงานข้อสรุปเป็นผลของ intervention ที่นิยามไว้บนชุดประเมินนั้น การอ้างว่า weight edit ช่วยให้ใช้งานจริงได้ต้องใช้ผลจากการแก้ weights จริงบน holdout และตรวจผลบนภาพปกติควบคู่กัน

## 7. สัญญาข้อมูลและ API

### สิ่งที่ต้องเก็บใน recipe และผลรัน

| ข้อมูล | ฟิลด์สำคัญ |
|---|---|
| Recipe identity | schema version, recipe ID, hypothesis, ผู้สร้างประเภท human/agent, hash ของสูตรที่ freeze |
| Model provenance | model ID, architecture revision, adapter version, parent checkpoint SHA-256 |
| Data provenance | dataset ID/hash, split, sample manifest hash, preprocessing revision/parameters และ seed เมื่อมีการสุ่ม |
| Intervention | target ID, mode, channel index, hook position หรือ tensor slice, operation, scale เมื่อเกี่ยวข้อง |
| Evaluation | evaluator version, metric IDs, slice IDs และ runtime metadata สำหรับตรวจสอบ reproducibility |
| Result | run ID, baseline run ID, recipe hash, status, metrics, sample counts, prediction differences และ error code เมื่อรันไม่สำเร็จ |

Recipe ที่ freeze ใช้ซ้ำได้และการแก้ค่าให้สร้าง revision ใหม่ ผลรันแต่ละครั้งมี run ID ใหม่แม้ใช้ recipe เดิม การตรวจว่าทำซ้ำได้ให้อ้าง recipe hash และ fingerprints ที่เกี่ยวข้องร่วมกับผลวัด

| การกระทำ | พฤติกรรมขั้นต่ำ | กรณีปฏิเสธที่ต้องทดสอบ |
|---|---|---|
| Inspect | คืนโมเดล โครงสร้างที่ยืนยันได้ target IDs และ allowed operations | model หรือ target ไม่อยู่ใน catalog |
| Propose | validate payload และคืน recipe ID หรือ field errors | channel เกินขอบเขต, mode ผิด, scale นอกช่วงหรือไม่ finite |
| Run | ตรวจสิทธิ์ fingerprint และงบ แล้วคืน run ID | dependency mismatch, โควตาหมด, holdout ไม่มีสิทธิ์ |
| Run status / Cancel | คืนสถานะ progress และคำสั่งยกเลิกตามสิทธิ์ | run ไม่มีอยู่, run จบแล้ว, run ของโปรเจกต์อื่น |
| Compare | คืนผล paired comparison จาก run ที่เสร็จ | run ยังไม่จบหรือ evaluation fingerprints ไม่เข้ากัน |
| Export checkpoint | สร้าง artifact หลังผู้ใช้ยืนยัน | activation run, run ไม่สำเร็จ, duplicate destination, Agent credential |

รายละเอียด HTTP paths เป็นงาน API specification ส่วน semantics ในตารางใช้ร่วมกันระหว่าง Desktop UI และ Agent API

## 8. สถานะ การกู้คืน และข้อกำหนดคุณภาพ

- **Lifecycle** ใช้สถานะ queued, running, succeeded, failed และ cancelled การ retry สร้าง run ID ใหม่และอ้าง run ก่อนหน้า เมื่อเปิดระบบใหม่ ให้ระบุรันที่ถูกตัดกลางทางว่า interrupted ภายใต้สถานะ failed และเปิดให้ retry ได้
- **Isolation** MVP ใช้หนึ่ง evaluation worker เป็นค่าเริ่มต้นและเก็บคำขอในคิว Working model แยกจาก checkpoint ต้นทาง ระบบต้องถอด hook และทิ้ง working state ทุกเส้นทางจบรัน รวมถึง worker failure
- **Artifact integrity** การเขียน recipe, metrics และ checkpoint ต้อง commit แบบอะตอมิกตามขอบเขต artifact ที่ผู้ใช้เปิดได้ หากเขียนล้มเหลว ผลครึ่งเดียวต้องไม่แสดงเป็นผลสำเร็จ
- **Reproducibility** บน runtime อ้างอิงเดียวกัน fixture identity ต้องได้ class predictions และ confusion counts ตรงกันทุกตัวอย่าง บันทึกเวอร์ชัน runtime และ device สำหรับเทียบข้ามเครื่อง กำหนด tolerance ของค่าทศนิยมถ้าต้องใช้ใน PoC ก่อนล็อกเกณฑ์ release
- **Agent budget** กำหนดเพดานจำนวนรันและ timeout ต่อรันจาก local config จองโควตาแบบอะตอมิกเมื่อ backend รับรัน รวมคำขอซ้ำด้วย idempotency key ต่อ session และแสดงงบคงเหลือ ค่าเริ่มต้นเสนอ 10 รันต่อ session เพื่อใช้ทดสอบ UX ก่อนปรับจากผล PoC
- **Local-first** UI/API bind ที่ loopback โดยปริยาย การรันโมเดลและประเมินผลไม่ต้องเชื่อม LLM provider ข้อมูลและ checkpoint อยู่ในเครื่อง ตัวอย่าง Agent ระบุข้อมูลที่ส่งออกไปยัง provider หากเปิดใช้งานการเชื่อมต่อ
- **Usability** มีหน้าสถานะสำหรับโหลด รันไม่สำเร็จ กลุ่มว่าง และผลพร้อมอ่านพร้อมกันหลายกลุ่ม ผู้ใช้เลือกดูผลด้วยคีย์บอร์ดได้และแยกผลดีขึ้น/แย่ลงได้จากข้อความหรือสัญลักษณ์
- **Resource use** วัดเวลารัน baseline, เวลาแสดงผล, peak RAM และขนาด artifacts บน fixture อ้างอิงใน PoC ให้ milestone ระบุชุดข้อมูลและ runtime ที่ใช้วัดก่อนตั้ง performance budget

## 9. หน้าจอหลัก

| หน้าจอ | ข้อมูลและปฏิสัมพันธ์ |
|---|---|
| Project | เปิด demo, ดู checkpoint/dataset fingerprints และเริ่ม baseline |
| Model Explorer | เลือก layer, channel และภาพ ดู shape, สถิติ kernel, activation map และตำแหน่ง hook |
| Experiment Editor | ตั้ง hypothesis, mode, target และ operation ดู preview ขอบเขตการแก้และข้อผิดพลาดรายช่อง |
| Run Comparison | ดูสถานะ ผลรวม confusion matrix ผลรายกลุ่ม และภาพพร้อมคำทำนายคู่ |
| History & Artifacts | เปิด recipe และผลเก่า ป้อนสูตรจาก JSON ส่งออกผล และขอสร้าง child checkpoint |
| Final Evaluation | ดูสูตรที่เลือกและขอบเขต holdout ยืนยันก่อนเริ่ม แล้วแสดงผลกับผล baseline คู่กัน |

## 10. เกณฑ์ตรวจรับก่อน release

| Gate | หลักฐานตรวจรับ |
|---|---|
| G1 · Quickstart | จากเครื่องสะอาดบน OS เป้าหมาย ติดตั้งและเปิด desktop app แล้วรัน demo จนเห็น comparison ทดสอบเปิด backend จาก CLI และรันผ่าน Agent โดยไม่เปิดหน้าต่างได้ด้วย |
| G2 · Mapping | Fixture ตรวจ hook channel และ kernel index ทุกเป้าหมายที่เปิดใน catalog รวม target แรกและสุดท้าย |
| G3 · Semantics | ทดสอบ zero/scale ทั้งสองโมด, scale = 1 ให้ prediction ตรง baseline และมี fixture ที่แสดงสองโหมดให้ผลต่างกันเมื่อมี bias |
| G4 · Isolation | ลำดับ baseline → edit → failure/cancel → baseline ให้ baseline predictions ตรงกัน และ hash ต้นทางคงเดิม |
| G5 · Metrics | ผล fixture ที่ทราบคำตอบให้ accuracy, confusion, delta และรายการที่เปลี่ยนตรงตามค่าที่คำนวณไว้ รวมกรณีกลุ่มว่าง |
| G6 · UI/API parity | รันสูตรเดียวกันผ่าน Desktop UI และ Agent API ด้วย fingerprints เดียวกัน ได้ prediction counts, metrics และรายการ sample IDs ตรงกัน |
| G7 · Export/replay | ส่งออก recipe แล้วเปิดใน LayerDelta instance ใหม่ได้ โหลด child checkpoint และได้ผลตรงกับ weight run ตรวจ tensor diff เทียบ parent |
| G8 · Error paths | ตรวจ malformed recipe, unsupported model, hash mismatch, quota exhausted, unauthorized export และ disk write failure |
| G9 · Holdout | ตรวจว่า source IDs ไม่ข้าม split และ Agent สำรวจไม่เห็น holdout ยืนยัน provenance ของผลรันสุดท้าย |
| G10 · Release package | มี desktop package สำหรับ OS เป้าหมายและวิธีติดตั้ง headless CLI พร้อม dependencies ระบุรุ่น คู่มือทดลองแรก ตัวอย่าง Agent คู่มือ adapter ที่มาของ artifacts และการตัดสินใจเรื่องสิทธิ์เผยแพร่ |

## 11. การประเมินคุณค่าหลัง PoC

| สมมติฐาน | วิธีวัดและตีความ |
|---|---|
| LayerDelta ช่วยให้คนวิเคราะห์เร็วขึ้น | ให้ผู้ใช้ทำโจทย์เดียวกันด้วย LayerDelta และสคริปต์อ้างอิง สลับลำดับการใช้ วัดเวลาถึงข้อสรุปที่ถูกต้อง งานที่ทำเสร็จ และการตรวจผลเสียข้ามกลุ่ม |
| ข้อมูลจากการแทรกแซงช่วย Agent ตั้งสมมติฐาน | ใช้ LM รุ่นเดียวกัน งานชุดเดียวกัน และงบเครื่องมือ/เวลารวมเท่ากัน เทียบกลุ่มที่เห็นผลรวมกับกลุ่มที่เห็นผลราย channel วัดความแม่นในการทำนายทิศทางผลของ intervention ที่ยังไม่เห็นและสัดส่วนสูตรที่รันได้ |
| Weight edit มีประโยชน์ต่อโจทย์ภาพเบลอ | เลือกสูตรจาก exploration แล้วล็อก candidate ก่อนเปิด holdout เปรียบเทียบกับ checkpoint เดิมและวิธีอ้างอิงที่เหมาะกับโจทย์ รายงานผล normal/blurred พร้อม N และจำนวนตัวอย่างที่ดีขึ้น/แย่ลง |

เสนอให้เก็บค่าตั้งต้นของเวลาและอัตราทำงานสำเร็จใน pilot ก่อนตั้งเป้าตัวเลข การพิสูจน์ความถูกต้องของเครื่องมือใช้ release gates ในข้อ 10 ส่วนผลวิจัยทั้งสามรายการใช้ตัดสินทิศทางหลัง pilot รวมถึงกรณีผลออกมาเป็นศูนย์หรือเป็นลบ

## 12. แผนส่งมอบ

| Milestone | งานหลัก | เกณฑ์ออกจาก milestone |
|---|---|---|
| M0 · Reference specification | กำหนด CNN, layer IDs, checkpoint, split manifests และ preprocessing | เปิด fixture อ้างอิงได้และล็อก fingerprints ที่ใช้ใน PoC |
| M1 · Runner PoC | สร้าง runner, adapter mapping, baseline, zero/scale และ paired comparison | รันจากเครื่องเดียวได้ครบสองโหมด ส่งผล JSON และผ่าน identity/isolation checks |
| M2 · Product alpha | เชื่อม desktop shell, headless CLI, API, queue, recipe replay และ export | ทำ workflow ตั้งแต่เปิด demo ถึง replay ผ่าน Desktop UI และ Agent API ได้ |
| M3 · Beta validation | ทดสอบ holdout, failure paths, quickstart และ pilot กับผู้ใช้ตัวอย่าง | ผ่าน G1–G9 และบันทึกปัญหาพร้อมผลวัดจาก pilot |
| M4 · Open-source v0.1 | เตรียม release package, คู่มือ และ issue templates | ผ่าน G10 และเจ้าของโครงการยืนยัน license กับสิทธิ์เผยแพร่ artifacts |

## 13. ความเสี่ยงและการตัดสินใจ

| ความเสี่ยง | การรับมือ |
|---|---|
| การแสดง graph ทำให้เข้าใจผิดว่ารองรับทุก operation | ใช้ mapping จาก adapter และแสดงตำแหน่งจริงของ hook กับ tensor slice |
| ผู้ใช้เหมารวมว่า channel เท่ากับคลาส | แสดงผลทุกคลาสและคู่ normal/blurred พร้อมตัวอย่างที่ดีขึ้นและแย่ลง |
| ปรับจนชนะเฉพาะ exploration | ตรึงการแบ่งข้อมูลตาม source ID บันทึกจำนวนรันและสูตรที่เลือกก่อนเปิด holdout |
| ผลเปลี่ยนจาก runtime หรือแหล่งข้อมูล | บันทึก fingerprints และใช้ runtime อ้างอิงสำหรับเกณฑ์ replay |
| Agent ใช้รันมากจนตามผลยาก | จำกัดโควตาและหนึ่งเป้าหมายต่อสูตร เก็บ provenance และแสดงงบคงเหลือ |
| สิทธิ์แจก checkpoint หรือข้อมูลยังไม่ชัด | ให้เจ้าของโครงการตรวจสิทธิ์แยกแต่ละ artifact และเตรียมวิธีจัดหาที่ได้รับอนุญาตก่อน release |

| ประเด็นที่ต้องยืนยัน | ข้อเสนอตั้งต้น | จุดตัดสินใจ |
|---|---|---|
| License | Apache-2.0 สำหรับโค้ด | ก่อนเปิด repository สาธารณะ |
| Reference CNN และ dataset | CNN สอง Conv blocks บน MNIST | M0 หลังยืนยันความเหมาะสมและที่มาของ artifacts |
| Blur severity และขนาด split | พารามิเตอร์คงที่ที่สร้าง failure slice ได้เพียงพอ | M0–M1 จากผล baseline และ sample counts |
| วิธีเลือก LM | รองรับตัวเรียกผ่าน API เดียว เลือก LM รุ่นเดียวต่อชุดทดลองเปรียบเทียบ | ก่อน Agent pilot |
| Quota, timeout และ performance targets | ใช้ค่า config และผล benchmark จาก PoC | ก่อน beta |

หลัง v0.1 ลำดับที่เสนอคือเพิ่ม Transformer ขนาดเล็กเพื่อทดสอบสัญญา adapter จากนั้นเพิ่ม OCR recognizer สำหรับข้อความที่ครอปแล้วพร้อม Character Error Rate และนำผล ablation ไปตั้งสมมติฐานสำหรับ LLM-Guided Dynamic Dropout ในงานฝึกโมเดลแยก โดยวัดผลของกติกา dropout ใหม่ใน training protocol ของงานนั้น
