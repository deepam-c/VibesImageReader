# CV Analytics Pro

Advanced computer vision analysis for clothing, accessories, and personal attributes with real-time Firebase integration.

## 🚀 **Primary Services**

### **Backend Service**: `backend/app_flask_async.py`
- **Flask + WebSocket** backend for image processing
- **Async support** with threading
- **Firebase integration** for data storage
- **CORS enabled** for frontend communication

### **Frontend**: Direct Firebase Connection
- **Real-time dashboard** with Firebase onSnapshot listeners
- **No backend dependency** for dashboard data
- **Hybrid connection** with WebSocket fallback support

---

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend        │    │   Firebase      │
│                 │    │                  │    │                 │
│ Dashboard ──────┼────┼──────────────────┼───▶│ Real-time Data  │
│ (Firebase       │    │ app_flask_async  │    │ (onSnapshot)    │
│  onSnapshot)    │    │                  │    │                 │
│                 │    │                  │    │                 │
│ Image Upload ───┼───▶│ CV Processing ───┼───▶│ Analysis Store  │
│                 │    │ (AI Analysis)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Dashboard Data Flow**:
1. **Frontend** connects directly to **Firebase** using `onSnapshot` listeners
2. **Real-time updates** when new analyses are added
3. **No backend calls** for dashboard data (eliminates quota issues)
4. **Consistent data** between dashboard and view-data pages

### **Image Analysis Flow**:
1. **Frontend** uploads image to **Backend** (`app_flask_async.py`)
2. **Backend** processes with **AI models** (face detection, demographics, clothing)
3. **Backend** stores results in **Firebase**
4. **Frontend dashboard** automatically updates via **Firebase listeners**

---

## 📊 **Features**

- ✅ **Real-time Dashboard** with Firebase onSnapshot
- ✅ **Computer Vision Analysis** (faces, demographics, clothing)
- ✅ **Direct Firebase Integration** (no backend dependency for dashboard)
- ✅ **Hybrid Connection Support** (WebSocket + HTTP fallback)
- ✅ **Average Confidence Tracking** from actual AI analysis
- ✅ **Responsive UI** with real-time status indicators

---

## 🔧 **Quick Start**

### Backend:
```bash
cd backend
python app_flask_async.py
```

### Frontend:
```bash
npm install
npm run dev
```

### Firebase:
- Configure `firebase-service-account.json`
- Database: Firestore with `cvAnalyses` collection

---

## 🎯 **Current Implementation Status**

✅ **Working**: Firebase direct dashboard connection  
✅ **Working**: Real-time data updates (442 people, 1326 clothing)  
✅ **Working**: Average confidence from real Firebase data  
✅ **Working**: Hybrid WebSocket/polling system  
✅ **Working**: Consistent data across all pages  

**Primary Service**: `backend/app_flask_async.py` (Flask + WebSocket + Firebase) 