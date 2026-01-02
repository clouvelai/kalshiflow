import React from 'react';

// Custom animated Signal Ribbon logo component
const SignalRibbonLogo = ({ className }) => {
  return (
    <svg 
      className={className}
      viewBox="0 0 48 48" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        {/* Gradient definition */}
        <linearGradient id="signalGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#22C55E">
            <animate attributeName="stop-color" 
              values="#22C55E;#10B981;#14B8A6;#22C55E" 
              dur="4s" 
              repeatCount="indefinite" />
          </stop>
          <stop offset="50%" stopColor="#10B981">
            <animate attributeName="stop-color" 
              values="#10B981;#14B8A6;#22C55E;#10B981" 
              dur="4s" 
              repeatCount="indefinite" />
          </stop>
          <stop offset="100%" stopColor="#14B8A6">
            <animate attributeName="stop-color" 
              values="#14B8A6;#22C55E;#10B981;#14B8A6" 
              dur="4s" 
              repeatCount="indefinite" />
          </stop>
        </linearGradient>
        
        {/* Glow filter */}
        <filter id="glow">
          <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
      </defs>
      
      {/* Animated wave path */}
      <path 
        d="M 4 24 Q 12 14 20 24 T 36 24 T 44 24"
        stroke="url(#signalGradient)"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
        filter="url(#glow)"
        opacity="0.9"
      >
        <animate 
          attributeName="d" 
          values="M 4 24 Q 12 14 20 24 T 36 24 T 44 24;
                  M 4 24 Q 12 34 20 24 T 36 24 T 44 24;
                  M 4 24 Q 12 24 20 24 T 36 34 T 44 24;
                  M 4 24 Q 12 14 20 24 T 36 24 T 44 24"
          dur="3s"
          repeatCount="indefinite"
        />
      </path>
      
      {/* Second wave for depth */}
      <path 
        d="M 4 24 Q 12 34 20 24 T 36 24 T 44 24"
        stroke="url(#signalGradient)"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
        opacity="0.5"
      >
        <animate 
          attributeName="d" 
          values="M 4 24 Q 12 34 20 24 T 36 24 T 44 24;
                  M 4 24 Q 12 24 20 24 T 36 14 T 44 24;
                  M 4 24 Q 12 14 20 24 T 36 24 T 44 24;
                  M 4 24 Q 12 34 20 24 T 36 24 T 44 24"
          dur="3s"
          begin="0.5s"
          repeatCount="indefinite"
        />
      </path>
      
      {/* Pulsing dots */}
      <circle cx="4" cy="24" r="2" fill="#22C55E">
        <animate attributeName="r" values="2;3;2" dur="2s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="1;0.6;1" dur="2s" repeatCount="indefinite"/>
      </circle>
      <circle cx="20" cy="24" r="2" fill="#10B981">
        <animate attributeName="r" values="2;3;2" dur="2s" begin="0.5s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="1;0.6;1" dur="2s" begin="0.5s" repeatCount="indefinite"/>
      </circle>
      <circle cx="36" cy="24" r="2" fill="#14B8A6">
        <animate attributeName="r" values="2;3;2" dur="2s" begin="1s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="1;0.6;1" dur="2s" begin="1s" repeatCount="indefinite"/>
      </circle>
      <circle cx="44" cy="24" r="2" fill="#22C55E">
        <animate attributeName="r" values="2;3;2" dur="2s" begin="1.5s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="1;0.6;1" dur="2s" begin="1.5s" repeatCount="indefinite"/>
      </circle>
    </svg>
  );
};

const Header = ({ connectionStatus, ...props }) => {
  return (
    <header 
      className="relative overflow-hidden bg-slate-950/95 backdrop-blur-2xl border-b border-slate-800/50" 
      {...props} 
      data-testid={props['data-testid'] || "header"}
    >
      {/* Multi-layered animated gradient backgrounds */}
      <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 via-emerald-500/5 to-teal-500/10 animate-gradient-xy"></div>
      <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-green-500/5 to-transparent animate-gradient-diagonal"></div>
      
      {/* Enhanced glow effects */}
      <div className="absolute top-0 left-1/4 -translate-x-1/2 w-96 h-64 bg-gradient-to-b from-green-500/20 to-transparent blur-3xl animate-pulse-slow"></div>
      <div className="absolute top-0 right-1/4 translate-x-1/2 w-96 h-64 bg-gradient-to-b from-teal-500/20 to-transparent blur-3xl animate-pulse-slow" style={{ animationDelay: '2s' }}></div>
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60rem] h-96 bg-gradient-radial from-green-500/10 to-transparent blur-3xl"></div>
      
      {/* Hero Content */}
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 sm:py-20 lg:py-24">
        <div className="flex flex-col items-center text-center space-y-8">
          {/* Animated Signal Ribbon Logo - Larger */}
          <div className="relative w-24 h-24 sm:w-28 sm:h-28 lg:w-32 lg:h-32 flex items-center justify-center transform transition-transform hover:scale-110 duration-500 group">
            <div className="absolute inset-0 bg-gradient-to-br from-green-500/30 to-teal-500/30 rounded-2xl blur-2xl group-hover:blur-3xl transition-all duration-500 animate-pulse-slow"></div>
            <div className="relative w-full h-full bg-gradient-to-br from-slate-900/90 to-slate-800/90 rounded-2xl border border-green-500/30 group-hover:border-green-500/50 transition-all duration-500 flex items-center justify-center shadow-2xl shadow-green-500/20">
              <SignalRibbonLogo className="w-20 h-20 sm:w-24 sm:h-24 lg:w-28 lg:h-28" />
            </div>
            {/* Rotating ring effect */}
            <div className="absolute inset-0 rounded-2xl border border-green-500/10 animate-spin-slow"></div>
          </div>
          
          {/* Hero Title and Subtitle */}
          <div className="space-y-4">
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight" data-testid="header-title">
              <span className="bg-gradient-to-r from-green-400 via-emerald-400 to-teal-400 bg-clip-text text-transparent animate-gradient-x inline-block transform hover:scale-105 transition-transform duration-300">
                KalshiFlow
              </span>
            </h1>
            <p className="text-slate-300 text-lg sm:text-xl lg:text-2xl font-medium tracking-wide max-w-2xl mx-auto leading-relaxed" data-testid="header-subtitle">
              Real-time Kalshi trading activity and insights
            </p>
            
            {/* Additional hero description */}
            <p className="text-slate-400 text-sm sm:text-base lg:text-lg max-w-3xl mx-auto mt-4 leading-relaxed">
              Monitor live market movements, track trading volumes, and discover trending prediction markets with real-time data streaming
            </p>
          </div>

          {/* Connection Status and Navigation - Integrated into hero */}
          <div className="flex items-center justify-center mt-6 space-x-4">
            <div className="flex items-center space-x-3 group px-6 py-3 bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800/50 hover:border-slate-700/50 transition-all duration-300" data-testid="connection-status">
              <div className="relative">
                <div className={`w-3 h-3 rounded-full ${
                  connectionStatus === 'connected' 
                    ? 'bg-green-400 shadow-lg shadow-green-400/50' 
                    : connectionStatus === 'connecting'
                    ? 'bg-amber-400 shadow-lg shadow-amber-400/50'
                    : 'bg-red-400 shadow-lg shadow-red-400/50'
                }`} data-testid="connection-indicator">
                  {connectionStatus === 'connected' && (
                    <>
                      <span className="absolute inset-0 rounded-full bg-green-400 animate-ping"></span>
                      <span className="absolute inset-0 rounded-full bg-green-400/50 animate-ping" style={{ animationDelay: '0.5s' }}></span>
                    </>
                  )}
                  {connectionStatus === 'connecting' && (
                    <span className="absolute inset-0 rounded-full bg-amber-400 animate-ping"></span>
                  )}
                </div>
              </div>
              
              <span className={`text-sm font-semibold tracking-wide transition-all duration-300 ${
                connectionStatus === 'connected'
                  ? 'text-green-400'
                  : connectionStatus === 'connecting'
                  ? 'text-amber-400'
                  : 'text-red-400'
              }`} data-testid="connection-status-text">
                {connectionStatus === 'connected' 
                  ? 'Live Data Stream Active' 
                  : connectionStatus === 'connecting'
                  ? 'Establishing Connection...'
                  : 'Connection Offline'}
              </span>
            </div>
            
            {/* Navigation Links */}
            <div className="flex items-center space-x-3">
              <a 
                href="/rl-trader" 
                className="flex items-center space-x-2 px-6 py-3 bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800/50 hover:border-green-500/50 hover:bg-slate-900/70 transition-all duration-300 group"
              >
                <svg className="w-4 h-4 text-green-400 group-hover:text-green-300 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span className="text-sm font-semibold text-slate-300 group-hover:text-green-400 transition-colors">
                  RL Trader v1.0
                </span>
              </a>
              
              <a 
                href="/v3-trader" 
                className="flex items-center space-x-2 px-6 py-3 bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800/50 hover:border-blue-500/50 hover:bg-slate-900/70 transition-all duration-300 group"
              >
                <svg className="w-4 h-4 text-blue-400 group-hover:text-blue-300 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span className="text-sm font-semibold text-slate-300 group-hover:text-blue-400 transition-colors">
                  V3 Console
                </span>
              </a>
            </div>
          </div>

          {/* Decorative elements */}
          <div className="absolute bottom-8 left-8 w-2 h-2 bg-green-400/50 rounded-full animate-float"></div>
          <div className="absolute bottom-12 right-12 w-3 h-3 bg-teal-400/50 rounded-full animate-float" style={{ animationDelay: '1s' }}></div>
          <div className="absolute top-20 right-20 w-2 h-2 bg-emerald-400/50 rounded-full animate-float" style={{ animationDelay: '2s' }}></div>
          <div className="absolute top-24 left-16 w-1.5 h-1.5 bg-green-400/30 rounded-full animate-float" style={{ animationDelay: '3s' }}></div>
        </div>
      </div>
      
      {/* Bottom gradient borders - enhanced */}
      <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/50 to-transparent"></div>
      <div className="absolute bottom-1 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent"></div>
      
      {/* CSS for animations */}
      <style>{`
        @keyframes gradient-x {
          0%, 100% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
        }
        
        @keyframes gradient-xy {
          0%, 100% {
            background-position: 0% 0%;
          }
          25% {
            background-position: 100% 0%;
          }
          50% {
            background-position: 100% 100%;
          }
          75% {
            background-position: 0% 100%;
          }
        }
        
        @keyframes gradient-diagonal {
          0%, 100% {
            background-position: 0% 0%;
          }
          50% {
            background-position: 100% 100%;
          }
        }
        
        @keyframes float {
          0%, 100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-10px);
          }
        }
        
        @keyframes spin-slow {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        
        @keyframes pulse-slow {
          0%, 100% {
            opacity: 0.5;
          }
          50% {
            opacity: 0.8;
          }
        }
        
        .animate-gradient-x {
          background-size: 200% 200%;
          animation: gradient-x 4s ease infinite;
        }
        
        .animate-gradient-xy {
          background-size: 400% 400%;
          animation: gradient-xy 15s ease infinite;
        }
        
        .animate-gradient-diagonal {
          background-size: 200% 200%;
          animation: gradient-diagonal 10s ease infinite;
        }
        
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        
        .animate-spin-slow {
          animation: spin-slow 20s linear infinite;
        }
        
        .animate-pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
        
        .bg-gradient-radial {
          background: radial-gradient(circle, var(--tw-gradient-stops));
        }
      `}</style>
    </header>
  );
};

export default Header;