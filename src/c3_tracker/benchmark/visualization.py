"""Post-inference overlays: labels never enter tracker inference."""
import cv2
from .schema import write_json


def render_overlay(video, frames, evaluations, output, fps):
    names = sorted({gt.person_id for item in evaluations.values() for gt,_ in item['gt']})
    labels = {name:f'GT{index+1}' for index,name in enumerate(names)}
    header = 62
    cap = cv2.VideoCapture(str(video.path))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].width, frames[0].height+header))
    import numpy as np
    def text(image,label,x,y,color):
        (width,_),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.4,1)
        x=max(0,min(x,image.shape[1]-width-3));y=max(header+12,min(y,image.shape[0]-4))
        cv2.rectangle(image,(x,y-13),(x+width+3,y+3),(18,18,18),-1)
        cv2.putText(image,label,(x+1,y),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    try:
        if not writer.isOpened():
            raise RuntimeError(f'Cannot initialize benchmark overlay: {output}')
        for prediction in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, prediction.frame_index)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode overlay frame {prediction.frame_index}')
            image = cv2.resize(image, (prediction.width, prediction.height))
            canvas = np.full((prediction.height+header,prediction.width,3),20,dtype=np.uint8)
            canvas[header:]=image
            image=canvas
            record = evaluations[prediction.frame_index]
            for gt, box in record['gt']:
                color = (255, 255, 0)
                x1,y1,x2,y2 = map(int,box)
                y1+=header;y2+=header
                designated = gt.person_id==record['row']['designated_person_id']
                missed = designated and record['row']['target_outcome'] in ('unassigned','wrong_person','false_positive','unmatched_unjudgeable')
                if missed:
                    color=(0,255,255)
                cv2.rectangle(image,(x1,y1),(x2,y2),color,1)
                text(image,labels[gt.person_id]+(' target' if designated else ''),x1,y1-4,color)
            for observation in prediction.observations:
                if not observation.observed or not observation.confirmed:
                    continue
                selected = observation.track_id == prediction.selected_track_id
                wrong = record['row']['target_outcome'] in ('wrong_person','false_positive')
                color = (0,0,255) if selected and wrong else ((0,230,0) if selected else (0,150,255))
                x1,y1,x2,y2 = map(int,observation.bbox)
                y1+=header;y2+=header
                cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
                label = ('TARGET ' if selected else 'ID ') + observation.track_id[:12]
                text(image,label,x1,y2+13,color)
            heading = f'frame {prediction.frame_index} | {prediction.timestamp_s:.3f}s'
            cv2.putText(image,heading,(6,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            target = 'Target: ' + record['row']['target_outcome'].replace('_',' ')
            cv2.putText(image,target,(6,34),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            events = 'GT cyan | pred orange | target green/red'
            if record['switches']:
                events = 'ID SWITCH: ' + ','.join(labels[name] for name in record['switches'])
            elif record['row']['target_outcome'] == 'unassigned' and record['row']['target_visible_gt'] == 1:
                events = 'MISSED DESIGNATED TARGET'
            elif not record['row']['evaluated']:
                events = 'UNKNOWN ANNOTATIONS - NOT SCORED'
            # Scale the event line to remain legible inside the smallest operating condition.
            event_width=cv2.getTextSize(events,cv2.FONT_HERSHEY_SIMPLEX,.34,1)[0][0]
            event_scale=.34*min(1,(prediction.width-12)/max(1,event_width))
            cv2.putText(image,events,(6,53),cv2.FONT_HERSHEY_SIMPLEX,event_scale,(255,255,255),1)
            writer.write(image)
    finally:
        writer.release()
        cap.release()
    write_json(output.with_suffix('.labels.json'),{'ground_truth_labels':{label:name for name,label in labels.items()},
                                                  'overlay_header_pixels':header,'prediction_coordinates':'unchanged; header added only in visualization'})
