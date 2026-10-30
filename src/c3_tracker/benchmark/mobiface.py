"""Verified MobiFace80 native annotations -> existing video-aligned common schema.

The supplied toolkit enumerates decoded clip JPEGs from zero. Native annotation
frames are clip-relative; metadata first/last frames refer to the original video.
Missing annotation rows remain unknown, while exactly four zeros mean absence.
"""
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
import re

from .schema import GT_FIELDS, GroundTruth, finite, validate_box, probe_video, fingerprint, write_csv, write_json

META_FIELDS = ('name','youtube_video_id','width','height','frame_rate','first_frame','last_frame')
ANNOT_FIELDS = ('frame','left','top','width','height')


def integer(value, name):
    if not re.fullmatch(r'\d+', str(value).strip()):
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


@dataclass(frozen=True)
class MobiFaceMetadata:
    name: str
    youtube_video_id: str
    width: int
    height: int
    frame_rate: float
    first_frame: int
    last_frame: int
    split: str

    @property
    def frame_count(self):
        return self.last_frame-self.first_frame+1


def read_metadata_file(path, split):
    records = {}
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != META_FIELDS:
            raise ValueError(f'{path}: expected native metadata columns {META_FIELDS}')
        for line,row in enumerate(reader,2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('missing or extra metadata field')
                name = row['name']
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',name) or name in records:
                    raise ValueError('invalid or duplicate sequence name')
                if not re.fullmatch(r'[A-Za-z0-9_-]{11}',row['youtube_video_id']):
                    raise ValueError('invalid YouTube video ID')
                values = {key:integer(row[key],key) for key in ('width','height','first_frame','last_frame')}
                fps = finite(row['frame_rate'],'frame_rate')
                if min(values['width'],values['height'])<1 or fps<=0 or values['last_frame']<values['first_frame']:
                    raise ValueError('invalid dimensions, FPS or inclusive original frame range')
                records[name] = MobiFaceMetadata(name,row['youtube_video_id'],values['width'],values['height'],fps,
                                                values['first_frame'],values['last_frame'],split)
            except (ValueError,TypeError) as exc:
                raise ValueError(f'{path}:{line}: {exc}') from exc
    if not records:
        raise ValueError(f'{path}: no metadata rows')
    return records


def read_metadata(data_root):
    root = Path(data_root)
    train = read_metadata_file(root/'mobiface80'/'train.meta.csv','train')
    test = read_metadata_file(root/'mobiface80'/'test.meta.csv','test')
    if train.keys() & test.keys():
        raise ValueError('Official train/test metadata must be disjoint')
    combined = {**train,**test}
    all_rows = read_metadata_file(root/'mobiface80.meta.csv','all')
    if all_rows.keys()!=combined.keys():
        raise ValueError('Combined metadata differs from official train/test union')
    for name,row in combined.items():
        if {**asdict(row),'split':'all'} != asdict(all_rows[name]):
            raise ValueError(f'Combined/split metadata mismatch for {name}')
    return combined


def read_annotations(path, metadata):
    records = {}
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ANNOT_FIELDS:
            raise ValueError(f'{path}: expected native annotation columns {ANNOT_FIELDS}')
        for line,row in enumerate(reader,2):
            try:
                if None in row or any(value is None or not value.strip() for value in row.values()):
                    raise ValueError('missing or extra annotation field; absent targets require four explicit zeros')
                index = integer(row['frame'],'frame')
                if index in records or not 0<=index<metadata.frame_count:
                    raise ValueError('duplicate or out-of-range clip-relative annotation frame')
                x,y,w,h = (finite(row[key],key) for key in ANNOT_FIELDS[1:])
                if (x,y,w,h)==(0,0,0,0):
                    records[index] = None
                else:
                    if w<=0 or h<=0:
                        raise ValueError('visible target needs positive width/height; absence is exactly 0,0,0,0')
                    records[index] = validate_box((x,y,x+w,y+h),metadata.width,metadata.height)
            except (ValueError,TypeError) as exc:
                raise ValueError(f'{path}:{line}: {exc}') from exc
    if not records:
        raise ValueError(f'{path}: annotation file is empty')
    return dict(sorted(records.items()))


def audit_annotations(data_root, output=None):
    root = Path(data_root)
    metadata = read_metadata(root)
    audit = {'dataset':'mobiface','scope':'native annotation parsing only; no video inference or accuracy claims',
             'frame_origin':'clip-relative zero-based; original first/last metadata are inclusive',
             'sequences':{},'annotation_rows':0,'visible_rows':0,'absent_rows':0,'missing_annotation_frames':0}
    for name,meta in sorted(metadata.items()):
        path = root/'mobiface80'/meta.split/(name+'.annot.csv')
        records = read_annotations(path,meta)
        absent = sum(box is None for box in records.values())
        item = {'metadata':asdict(meta),'annotations_sha256':fingerprint(path),'rows':len(records),
                'absent_rows':absent,'visible_rows':len(records)-absent,
                'missing_annotation_frames':meta.frame_count-len(records),
                'first_listed_clip_frame':min(records),'last_listed_clip_frame':max(records)}
        audit['sequences'][name] = item
        for key in ('visible_rows','absent_rows','missing_annotation_frames'):
            audit[key] += item[key]
        audit['annotation_rows'] += len(records)
    audit['split_counts'] = {split:sum(meta.split==split for meta in metadata.values()) for split in ('train','test')}
    if output is not None:
        output = Path(output)
        if output.exists():
            raise ValueError('Will not overwrite annotation audit')
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(output,audit)
    return audit


def prepare_mobiface(manifest_path, output):
    manifest_path,output = Path(manifest_path).resolve(),Path(output).resolve()
    source = json.loads(manifest_path.read_text(encoding='utf-8'))
    if source.get('dataset')!='mobiface' or source.get('split') not in ('train','test') or not source.get('videos'):
        raise ValueError('MobiFace raw manifest needs dataset=mobiface, one official split=train/test, and videos')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preparation output must be fresh')
    root = (manifest_path.parent/source.get('data_root','.')).resolve()
    metadata = read_metadata(root)
    pending,seen = [],set()
    for entry in source['videos']:
        name = entry['video_id']
        if name in seen or name not in metadata:
            raise ValueError(f'Unknown/duplicate MobiFace sequence {name}')
        seen.add(name)
        meta = metadata[name]
        if meta.split!=source['split']:
            raise ValueError('Sequence does not belong to the requested official split')
        if entry.get('cfr') is not True or not entry.get('alignment_provenance'):
            raise ValueError('Require explicit cfr=true and alignment_provenance auditing the supplied media')
        layout = entry.get('video_layout')
        if layout not in ('clip','full_source'):
            raise ValueError('video_layout must explicitly be clip or full_source; offsets are never inferred')
        original_first = meta.first_frame if layout=='clip' else 0
        video = probe_video(manifest_path.parent/entry['video_path'],name,original_first/meta.frame_rate)
        if (video.width,video.height)!=(meta.width,meta.height) or abs(video.fps-meta.frame_rate)>1e-4:
            raise ValueError('Video dimensions/FPS differ from native metadata; verify media version/CFR conversion')
        if layout=='clip' and video.frame_count!=meta.frame_count:
            raise ValueError('Clip frame count differs from inclusive original metadata range; check trimming/offset')
        if layout=='full_source' and video.frame_count<=meta.last_frame:
            raise ValueError('Full source does not contain the original annotated frame range')
        annotation = root/'mobiface80'/meta.split/(name+'.annot.csv')
        records = read_annotations(annotation,meta)
        offset = meta.first_frame-original_first
        normalized = [GroundTruth(name,index+offset,(index+offset)/video.fps,'target',box,box is not None)
                      for index,box in records.items()]
        item = {'video_id':name,'video_path':str(video.path),'annotations_path':f'{name}/annotations.csv',
                'video_start_timestamp_s':video.video_start_timestamp_s,'cfr':True,'split':meta.split,
                'coverage':'partial','identity_scope':'track_local','identity_annotations_complete':False,
                'absence_annotations_complete':False,'annotation_type':'single_target','designated_person_id':'target',
                'start_frame':offset,'end_frame_exclusive':offset+meta.frame_count,
                'native_metadata':asdict(meta),'native_annotations_sha256':fingerprint(annotation),
                'native_annotation_path':str(annotation),'video_sha256':fingerprint(video.path),
                'missing_annotation_frames':meta.frame_count-len(records),
                'video_layout':layout,'alignment_provenance':entry['alignment_provenance'],
                'frame_origin':'native zero-based clip-relative; source index = native frame + original first - media first',
                'identity_note':'One annotated target; bystanders not exhaustive; no independently verified physical reacquisition'}
        pending.append((item,normalized))
    output.mkdir(parents=True,exist_ok=True)
    for item,rows in pending:
        path = output/item['annotations_path'];path.parent.mkdir(parents=True,exist_ok=True)
        write_csv(path,GT_FIELDS,(row.row() for row in rows));item['annotations_sha256']=fingerprint(path)
    result = {'schema_version':1,'dataset':'mobiface','split':source['split'],'release':'provided-MobiFace80-Drive',
              'source_manifest_sha256':fingerprint(manifest_path),'videos':[item for item,_ in pending]}
    write_json(output/'manifest.json',result)
    return result
