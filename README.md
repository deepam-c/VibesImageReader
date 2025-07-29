# Camera Capture App

A professional and responsive Next.js web application for capturing high-quality images from device cameras. Works seamlessly on both desktop and mobile devices with an elegant, modern UI.

## Features

- **🎥 Real-time Camera Preview**: Live webcam feed with high-quality video streaming
- **📸 Image Capture**: Capture high-resolution images with a single click
- **📱 Responsive Design**: Optimized for laptops, tablets, and mobile phones
- **🎨 Professional UI**: Elegant color scheme with smooth animations and transitions
- **💾 Download Images**: Save captured images directly to your device
- **🔄 Retake Functionality**: Easy retake option for better shots
- **🔒 Privacy-Focused**: All image processing happens locally in your browser

## Technology Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Heroicons** - Beautiful SVG icons
- **WebRTC** - Browser camera API integration

## Getting Started

### Prerequisites

- Node.js 18+ 
- Modern web browser with camera support
- Camera permissions enabled

### Installation

1. **Clone or download the project**
   ```bash
   cd camera-capture-app
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Run the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

## Usage

### Basic Usage

1. **Access the Application**
   - Open the app in your web browser
   - Grant camera permissions when prompted

2. **Navigate to Camera Capture**
   - Use the sidebar to navigate to "Capture Image"
   - Or click "Start Capturing" from the home page

3. **Capture Images**
   - Click "Start Camera" to activate your webcam
   - Position yourself in the camera preview
   - Click "Capture Image" to take a photo
   - Download or retake as needed

### Mobile Usage

- The app is fully responsive and works on mobile devices
- Use the hamburger menu to access navigation on smaller screens
- Camera functionality works with both front and rear cameras

## Browser Compatibility

The app works on all modern browsers that support:
- WebRTC and getUserMedia API
- ES6+ JavaScript features
- CSS Grid and Flexbox

**Supported Browsers:**
- Chrome 63+
- Firefox 60+
- Safari 11+
- Edge 79+

## Privacy & Security

- **No data transmission**: All image processing happens locally
- **No image storage**: Images are only stored temporarily in browser memory
- **User control**: Camera access requires explicit user permission
- **Local downloads**: Images are downloaded directly to your device

## Troubleshooting

### Camera Not Working

1. **Check Permissions**
   - Ensure camera permissions are granted in browser settings
   - Look for camera icon in address bar

2. **HTTPS Required**
   - Camera API requires HTTPS in production
   - Use `localhost` for development

3. **Browser Support**
   - Update to the latest browser version
   - Try a different browser if issues persist

### Performance Issues

- **Large Images**: Captured images are high-resolution by default
- **Older Devices**: May experience slower performance
- **Memory**: Refresh page if experiencing memory issues

## Development

### Project Structure

```
camera-capture-app/
├── app/                    # Next.js App Router
│   ├── capture/           # Camera capture page
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx          # Home page
├── components/            # React components
│   ├── CameraCapture.tsx  # Main camera component
│   └── Sidebar.tsx       # Navigation sidebar
├── public/               # Static assets
└── README.md            # This file
```

### Key Components

- **CameraCapture**: Handles webcam integration and image capture
- **Sidebar**: Responsive navigation component
- **Layout**: Root layout with sidebar integration

### Customization

- **Colors**: Modify CSS variables in `globals.css`
- **Image Quality**: Adjust `toDataURL` quality in `CameraCapture.tsx`
- **Camera Settings**: Modify constraints in `getUserMedia` call

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Support

For issues or questions:
- Check the troubleshooting section
- Review browser console for errors
- Ensure proper camera permissions

---

**Built with ❤️ using Next.js and modern web technologies** 