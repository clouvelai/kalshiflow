import React, { useState } from 'react';

const Contact = () => {
  const [emailCopied, setEmailCopied] = useState(false);

  const handleEmailClick = () => {
    const email = 'samuelmacarthurclark@gmail.com';
    navigator.clipboard.writeText(email).then(() => {
      setEmailCopied(true);
      setTimeout(() => setEmailCopied(false), 2000);
    }).catch(() => {
      // Fallback to mailto if clipboard fails
      window.location.href = `mailto:${email}`;
    });
  };

  const handlePortfolioClick = () => {
    window.open('https://www.samuelclark.dev/', '_blank', 'noopener,noreferrer');
  };

  const handleLinkedInClick = () => {
    window.open('https://www.linkedin.com/in/samclark77/', '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden h-fit">
      {/* Contact Header */}
      <div className="bg-gradient-to-r from-purple-600 to-purple-700 px-4 sm:px-6 py-6 text-center">
        <h2 className="text-xl sm:text-2xl font-bold text-white mb-2">
          Contact & About
        </h2>
        <p className="text-purple-100 text-sm">
          Get in touch with the developer
        </p>
      </div>

      {/* Contact Content */}
      <div className="p-4 sm:p-6 space-y-6">
        {/* Developer Introduction */}
        <div className="text-center">
          <h3 className="text-lg sm:text-xl font-semibold text-gray-900 mb-3">
            Samuel Clark
          </h3>
          <p className="text-gray-600 leading-relaxed mb-4 text-sm sm:text-base">
            Full-stack engineer passionate about real-time data visualization and financial technology. 
            This project showcases modern web architecture optimized for high-frequency trading data.
          </p>
        </div>

        {/* Contact Methods */}
        <div className="space-y-4">
          {/* Name */}
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-900">
              Name: Samuel Clark
            </div>
          </div>

          {/* Email */}
          <div className="group">
            <button
              onClick={handleEmailClick}
              className="w-full flex items-center justify-center gap-3 p-4 bg-gray-50 hover:bg-blue-50 border border-gray-200 hover:border-blue-300 rounded-lg transition-all duration-200"
              title="Click to copy email address"
            >
              <svg className="w-5 h-5 text-gray-500 group-hover:text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 7.89a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <div className="text-left flex-1">
                <div className="font-semibold text-gray-900 group-hover:text-blue-700">
                  Email: samuelmacarthurclark@gmail.com
                </div>
                <div className="text-sm text-gray-500 group-hover:text-blue-500">
                  {emailCopied ? 'Email copied to clipboard!' : 'Click to copy email address'}
                </div>
              </div>
            </button>
          </div>

          {/* Website */}
          <div className="group">
            <button
              onClick={handlePortfolioClick}
              className="w-full flex items-center justify-center gap-3 p-4 bg-gray-50 hover:bg-purple-50 border border-gray-200 hover:border-purple-300 rounded-lg transition-all duration-200"
            >
              <svg className="w-5 h-5 text-gray-500 group-hover:text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9v-9m0-9v9m0 9c-5 0-9-4-9-9s4-9 9-9" />
              </svg>
              <div className="text-left flex-1">
                <div className="font-semibold text-gray-900 group-hover:text-purple-700">
                  Website: https://www.samuelclark.dev/
                </div>
                <div className="text-sm text-gray-500 group-hover:text-purple-500">
                  View portfolio and other projects
                </div>
              </div>
              <svg className="w-4 h-4 text-gray-400 group-hover:text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </button>
          </div>

          {/* LinkedIn */}
          <div className="group">
            <button
              onClick={handleLinkedInClick}
              className="w-full flex items-center justify-center gap-3 p-4 bg-gray-50 hover:bg-blue-50 border border-gray-200 hover:border-blue-300 rounded-lg transition-all duration-200"
            >
              <svg className="w-5 h-5 text-gray-500 group-hover:text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8a6 6 0 016 6v7h-4v-7a2 2 0 00-2-2 2 2 0 00-2 2v7h-4v-7a6 6 0 016-6zM2 9h4v12H2z" />
                <circle cx="4" cy="4" r="2" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} />
              </svg>
              <div className="text-left flex-1">
                <div className="font-semibold text-gray-900 group-hover:text-blue-700">
                  LinkedIn: https://www.linkedin.com/in/samclark77/
                </div>
                <div className="text-sm text-gray-500 group-hover:text-blue-500">
                  Connect on LinkedIn
                </div>
              </div>
              <svg className="w-4 h-4 text-gray-400 group-hover:text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </button>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Contact;