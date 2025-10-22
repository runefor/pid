# Train vs Test 클래스 불균형 종합 분석 리포트

## Train 데이터셋 불균형 지표
- **총 클래스 수**: 105
- **최소 객체 수**: 1
- **최대 객체 수**: 14,527
- **평균 객체 수**: 1,594.43
- **객체 수 중앙값**: 588.00
- **최대/최소 클래스 비율**: 14,527.00
- **데이터의 50%를 차지하는 상위 클래스 수**: 54
- **데이터의 80%를 차지하는 상위 클래스 수**: 97
- **데이터의 95%를 차지하는 상위 클래스 수**: 104

## Test 데이터셋 불균형 지표
- **총 클래스 수**: 87
- **최소 객체 수**: 1
- **최대 객체 수**: 1,891
- **평균 객체 수**: 172.90
- **객체 수 중앙값**: 57.00
- **최대/최소 클래스 비율**: 1,891.00
- **데이터의 50%를 차지하는 상위 클래스 수**: 36
- **데이터의 80%를 차지하는 상위 클래스 수**: 63
- **데이터의 95%를 차지하는 상위 클래스 수**: 81

## 데이터셋 간 클래스 비교
- **Train 데이터셋에만 존재하는 클래스**: 19개
  - `Equipments@Eye Washer`, `Equipments@Spray Nozzle`, `General@Fire Damper`, `Instruments@Pressure Alarm`, `Instruments@Temperature Indicator Alarm`, `Instruments@Vibration Indicator`, `Piping Accessories@Compensator`, `Piping Accessories@Diffuser`, `Piping Accessories@Duplex Strainer`, `Piping Accessories@Ejector/Injector`, `Piping Accessories@Moisture Eliminator`, `Piping Accessories@Plug`, `Piping Accessories@Quick Connect`, `Piping Accessories@Steam Trap`, `Piping Accessories@Swivel Joint`, `Valves@ESD 3-Way Valve`, `Valves@Hydraulic Operated Butterfly Valve`, `Valves@Pneumatic Actuated Ball Valve`, `Valves@Solenoid Operated 3-Way Valve`
- **Test 데이터셋에만 존재하는 클래스**: 1개
  - `Piping Accessories@Vortex Eductor Nozzle`
