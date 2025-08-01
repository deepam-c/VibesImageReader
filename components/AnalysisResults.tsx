'use client'

import React from 'react'

interface AnalysisResultsProps {
  results: any
  isVisible?: boolean
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ results, isVisible = true }) => {
  if (!isVisible || !results?.success) {
    return null
  }

  const { detection_summary, people = [], scene_analysis } = results

  return (
    <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
        <svg className="w-6 h-6 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        Computer Vision Analysis
      </h3>

      {/* Detection Summary */}
      <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-blue-600">{detection_summary?.total_people_detected || 0}</div>
          <div className="text-sm text-gray-600">People Detected</div>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-green-600">{detection_summary?.faces_detected || 0}</div>
          <div className="text-sm text-gray-600">Faces Analyzed</div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-2xl font-bold text-purple-600">{detection_summary?.poses_detected || 0}</div>
          <div className="text-sm text-gray-600">Poses Detected</div>
        </div>
      </div>

      {/* People Analysis */}
      {people.length > 0 && (
        <div className="space-y-6">
          <h4 className="text-lg font-semibold text-gray-800">Individual Analysis</h4>
          {people.map((person: any, index: number) => (
            <PersonAnalysis key={index} person={person} />
          ))}
        </div>
      )}

      {/* Scene Analysis */}
      {scene_analysis && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="text-lg font-semibold text-gray-800 mb-2">Scene Analysis</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <span className="font-medium text-gray-600">Scene Type:</span>
              <div className="capitalize">{scene_analysis.scene_type?.replace('_', ' ')}</div>
            </div>
            <div>
              <span className="font-medium text-gray-600">Image Quality:</span>
              <div className="capitalize">{scene_analysis.image_quality}</div>
            </div>
            <div>
              <span className="font-medium text-gray-600">Lighting:</span>
              <div className="capitalize">{scene_analysis.lighting_conditions}</div>
            </div>
            <div>
              <span className="font-medium text-gray-600">Analysis Time:</span>
              <div>{results.metadata?.processing_time_ms || 'N/A'}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface PersonAnalysisProps {
  person: any
}

const PersonAnalysis: React.FC<PersonAnalysisProps> = ({ person }) => {
  const { demographics, physical_attributes, appearance, pose_analysis } = person

  return (
    <div className="border border-gray-200 rounded-lg p-5 bg-gradient-to-r from-gray-50 to-white">
      <div className="flex justify-between items-start mb-4">
        <h5 className="text-lg font-medium text-gray-900">
          Person {person.person_id + 1}
        </h5>
        <div className="text-sm text-gray-500">
          Confidence: {Math.round((person.confidence_metrics?.overall_confidence || 0) * 100)}%
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Demographics */}
        <div className="space-y-3">
          <h6 className="font-medium text-gray-800 flex items-center">
            <svg className="w-4 h-4 mr-2 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
            </svg>
            Demographics
          </h6>
          <div className="ml-6 space-y-2 text-sm">
            {demographics?.age && (
              <div className="flex justify-between">
                <span className="text-gray-600">Age:</span>
                <span className="font-medium">
                  {demographics.age.estimated_age !== 'unknown' ? 
                    `${demographics.age.estimated_age} years (${demographics.age.age_range?.replace('_', ' ')})` :
                    'Unknown'
                  }
                </span>
              </div>
            )}
            {demographics?.gender && (
              <div className="flex justify-between">
                <span className="text-gray-600">Gender:</span>
                <span className="font-medium capitalize">
                  {demographics.gender.prediction}
                  <span className="text-xs text-gray-500 ml-1">
                    ({demographics.gender.confidence})
                  </span>
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Physical Attributes */}
        <div className="space-y-3">
          <h6 className="font-medium text-gray-800 flex items-center">
            <svg className="w-4 h-4 mr-2 text-green-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zm0 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V8zm0 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1v-2z" clipRule="evenodd" />
            </svg>
            Physical Attributes
          </h6>
          <div className="ml-6 space-y-2 text-sm">
            {physical_attributes?.hair?.detected && (
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">Hair Color:</span>
                  <span className="font-medium capitalize">{physical_attributes.hair.color}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Hair Style:</span>
                  <span className="font-medium capitalize">{physical_attributes.hair.style}</span>
                </div>
              </div>
            )}
            {physical_attributes?.facial_features && (
              <div className="flex justify-between">
                <span className="text-gray-600">Face Shape:</span>
                <span className="font-medium capitalize">{physical_attributes.facial_features.face_shape}</span>
              </div>
            )}
          </div>
        </div>

        {/* Appearance */}
        <div className="space-y-3">
          <h6 className="font-medium text-gray-800 flex items-center">
            <svg className="w-4 h-4 mr-2 text-purple-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M2 5a2 2 0 012-2h8a2 2 0 012 2v10a2 2 0 002 2H4a2 2 0 01-2-2V5zm3 1h6v4H5V6zm6 6H5v2h6v-2z" clipRule="evenodd" />
            </svg>
            Appearance
          </h6>
          <div className="ml-6 space-y-2 text-sm">
            {appearance?.overall_style && (
              <div className="flex justify-between">
                <span className="text-gray-600">Style:</span>
                <span className="font-medium capitalize">{appearance.overall_style}</span>
              </div>
            )}
                         {appearance?.clothing?.detected_items && appearance.clothing.detected_items.length > 0 && (
               <div className="space-y-1">
                 <div className="flex justify-between">
                   <span className="text-gray-600">Clothing:</span>
                   <div className="flex flex-wrap gap-1">
                     {appearance.clothing.detected_items.slice(0, 3).map((item: string, idx: number) => (
                       <span key={idx} className="px-2 py-1 bg-blue-100 text-xs rounded capitalize">
                         {item.replace('_', ' ')}
                       </span>
                     ))}
                   </div>
                 </div>
                 {appearance.clothing.patterns && appearance.clothing.patterns.length > 0 && (
                   <div className="flex justify-between">
                     <span className="text-gray-600">Patterns:</span>
                     <div className="flex flex-wrap gap-1">
                       {appearance.clothing.patterns.slice(0, 2).map((pattern: string, idx: number) => (
                         <span key={idx} className="px-2 py-1 bg-green-100 text-xs rounded capitalize">
                           {pattern.replace('_', ' ')}
                         </span>
                       ))}
                     </div>
                   </div>
                 )}
                 <div className="flex justify-between">
                   <span className="text-gray-600">Colors:</span>
                   <div className="flex space-x-1">
                     {appearance.clothing.dominant_colors?.slice(0, 3).map((color: string, idx: number) => (
                       <span key={idx} className="px-2 py-1 bg-gray-100 text-xs rounded capitalize">
                         {color}
                       </span>
                     ))}
                   </div>
                 </div>
               </div>
             )}
             
             {/* Accessories Section */}
             {appearance?.accessories && appearance.accessories.length > 0 && (
               <div className="flex justify-between">
                 <span className="text-gray-600">Accessories:</span>
                 <div className="flex flex-wrap gap-1">
                   {appearance.accessories.slice(0, 4).map((accessory: string, idx: number) => (
                     <span key={idx} className="px-2 py-1 bg-purple-100 text-xs rounded capitalize">
                       {accessory.replace('_', ' ')}
                     </span>
                   ))}
                 </div>
               </div>
             )}
             
             {/* Outfit Formality */}
             {appearance?.outfit_formality && (
               <div className="flex justify-between">
                 <span className="text-gray-600">Formality:</span>
                 <span className="font-medium capitalize text-indigo-600">
                   {appearance.outfit_formality.replace('_', ' ')}
                 </span>
               </div>
             )}
          </div>
        </div>

        {/* Pose Analysis */}
        <div className="space-y-3">
          <h6 className="font-medium text-gray-800 flex items-center">
            <svg className="w-4 h-4 mr-2 text-orange-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 2a4 4 0 100 8 4 4 0 000-8zM8 14a2 2 0 00-2 2v1a1 1 0 001 1h6a1 1 0 001-1v-1a2 2 0 00-2-2H8z" clipRule="evenodd" />
            </svg>
            Pose & Activity
          </h6>
          <div className="ml-6 space-y-2 text-sm">
            {pose_analysis?.pose_detected && (
              <>
                <div className="flex justify-between">
                  <span className="text-gray-600">Position:</span>
                  <span className="font-medium capitalize">{pose_analysis.body_position}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Activity:</span>
                  <span className="font-medium capitalize">{pose_analysis.activity}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Pose Confidence:</span>
                  <span className="font-medium">{Math.round((pose_analysis.pose_confidence || 0) * 100)}%</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default AnalysisResults 