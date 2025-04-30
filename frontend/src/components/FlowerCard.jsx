import React from 'react';

const getGradientClass = (color) => {
    const colorMap = {
      'red': 'from-red-50 to-rose-100',
      'pink': 'from-pink-50 to-rose-100',
      'yellow': 'from-amber-50 to-yellow-100',
      'blue': 'from-blue-50 to-indigo-100',
      'purple': 'from-purple-50 to-fuchsia-100',
      'white': 'from-gray-50 to-slate-100',
      'orange': 'from-orange-50 to-red-100',
      'green': 'from-green-50 to-emerald-100',
    };
    
    return colorMap[color.toLowerCase()] || 'from-gray-50 to-slate-100';
};

const FlowerCard = ({ flower, selected, onClick }) => {
    const { id, name, size, color, image_url, description } = flower; 
    const gradientClass = getGradientClass(color);

    return (
        <div 
            className={`rounded-xl overflow-hidden shadow-md transition-all duration-300 transform cursor-pointer
                ${selected ? 'scale-105 ring-2 ring-offset-2 ring-indigo-500' : 'hover:scale-102'}`}
            onClick={() => onClick(flower)}
        >
            <div className={`bg-gradient-to-br ${gradientClass} p-4 h-full flex flex-col`}>
                <div className="flex justify-between items-start mb-3">
                    <span className="bg-white/80 text-gray-800 text-xs font-medium px-2.5 py-1 rounded-full">
                        {size}
                    </span>
                    <span className="bg-white/80 text-gray-800 text-xs font-medium px-2.5 py-1 rounded-full">
                        {color}
                    </span>
                </div>
                
                <div className="flex justify-center mb-3">
                    <img 
                        src={image_url} 
                        alt={name} 
                        className="rounded-lg h-40 w-full object-cover"
                        onError={(e) => {
                            e.target.onerror = null;
                            e.target.src = '/placeholder-flower.jpg';
                        }}
                    />
                </div>
                
                <div className="text-center">
                    <h3 className="text-gray-800 font-semibold">{name}</h3>
                    
                    {selected && (
                        <div className="mt-2 text-xs text-gray-600 line-clamp-3">
                            {description}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default FlowerCard;

